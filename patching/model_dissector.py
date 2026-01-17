from collections import defaultdict
import torch

class ModelDissector:
    def __init__(self, model, tokenizer, max_new_tokens=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

        self.residual_cache = defaultdict(lambda: defaultdict(list))
        self.attention_cache = defaultdict(lambda: defaultdict(list))
        self.mlp_cache = defaultdict(lambda: defaultdict(list))

        # hook handles
        self.caching_hook_handles = []
        self.patching_hook_handles = []

        # indices for caching hooks to keep track of the tokens generated
        self.residual_token_idx = 0
        self.attention_token_idx = 0
        self.mlp_token_idx = 0

        # index for patching hooks to keep track of the token being patched
        self.patch_token_idx = 0

        # length in order to not patch system message tokens
        self.system_message_length = None

        # logits for calculating average logit differences    
        self.clean_logits = None
        
        self.clean_output = []

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    # helper function to get token positions and full answer, useful for determining at which positions to patch
    def get_token_positions(self, user_message, system_message):
        positions = {}
        input = self.tokenizer.apply_chat_template(
            [system_message, user_message], return_tensors="pt"
        ).to(self.model.device)
        outputs = self.__generate(input)
        for token_idx, token in enumerate(outputs.sequences[:, input.shape[1] :][0]):
            positions[token_idx] = self.tokenizer.decode(
                token, skip_special_tokens=False
            )
        complete_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=False
        )
        return positions, complete_answer

    def __generate(self, input):
        attention_mask = torch.ones_like(input)
        with torch.no_grad():
            outputs = self.model.generate(
                input,
                max_new_tokens=self.max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                output_logits=True,
                output_scores=True,
                do_sample=False,
            )
        return outputs

    def __register_caching_hooks(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            def forward_residual_hook(module, input, output, layer_idx=layer_idx):
                if layer_idx == 0:
                    self.residual_token_idx += 1
                self.residual_cache[self.residual_token_idx - 1][layer_idx] = output[0][0]

            def forward_attention_hook(module, input, output, layer_idx=layer_idx):
                if layer_idx == 0:
                    self.attention_token_idx += 1
                self.attention_cache[self.attention_token_idx - 1][layer_idx] = output[
                    0
                ][0]

            def forward_mlp_hook(module, input, output, layer_idx=layer_idx):
                if layer_idx == 0:
                    self.mlp_token_idx += 1
                self.mlp_cache[self.mlp_token_idx - 1][layer_idx] = output[0]

            self.caching_hook_handles.append(
                layer.register_forward_hook(forward_residual_hook)
            )
            self.caching_hook_handles.append(
                layer.self_attn.register_forward_hook(forward_attention_hook)
            )
            self.caching_hook_handles.append(
                layer.mlp.register_forward_hook(forward_mlp_hook)
            )

    def __remove_caching_hooks(self):
        for handle in self.caching_hook_handles:
            handle.remove()
        self.caching_hook_handles = []
        self.residual_token_idx = 0
        self.attention_token_idx = 0
        self.mlp_token_idx = 0

    def generate_cache(self, system_message, user_message):
        self.system_message_length = len(
            self.tokenizer.apply_chat_template(
                [system_message], return_tensors="pt"
            ).to(self.model.device)[0]
        )
        clean_input = self.tokenizer.apply_chat_template(
            [system_message, user_message], return_tensors="pt"
        ).to(self.model.device)
        self.__register_caching_hooks()
        clean_outputs = self.__generate(clean_input)
        self.__remove_caching_hooks()
        self.clean_logits = clean_outputs.logits
        for token in clean_outputs.sequences[:, :][0]:
            self.clean_output.append(
                self.tokenizer.decode(token, skip_special_tokens=False)
            )

        return self.residual_cache, self.attention_cache, self.mlp_cache, clean_outputs

    def __register_residual_patching_hook(self, token_idx, layer_idx, seq_idx):
        def forward_layer_hook(
            module,
            input,
            output,
            token_idx=token_idx,
            layer_idx=layer_idx,
            seq_idx=seq_idx,
        ):
            if self.patch_token_idx == token_idx:
                output[0][0][seq_idx].copy_(
                    self.residual_cache[token_idx][layer_idx][seq_idx]
                )
            self.patch_token_idx += 1
            return output

        self.patching_hook_handles.append(
            self.model.model.layers[layer_idx].register_forward_hook(forward_layer_hook)
        )

    def __register_attention_patching_hook(self, token_idx, layer_idx, seq_idx):
        def forward_layer_hook(
            module,
            input,
            output,
            token_idx=token_idx,
            layer_idx=layer_idx,
            seq_idx=seq_idx,
        ):
            if token_idx == self.patch_token_idx:
                output[0][0][seq_idx].copy_(
                    self.attention_cache[token_idx][layer_idx][seq_idx]
                )
            self.patch_token_idx += 1
            return output

        self.patching_hook_handles.append(
            self.model.model.layers[layer_idx].self_attn.register_forward_hook(
                forward_layer_hook
            )
        )

    def __register_mlp_patching_hook(self, token_idx, layer_idx, seq_idx):
        def forward_layer_hook(
            module,
            input,
            output,
            token_idx=token_idx,
            layer_idx=layer_idx,
            seq_idx=seq_idx,
        ):
            if token_idx == self.patch_token_idx:
                output[0][seq_idx].copy_(self.mlp_cache[token_idx][layer_idx][seq_idx])
            self.patch_token_idx += 1
            return output

        self.patching_hook_handles.append(
            self.model.model.layers[layer_idx].mlp.register_forward_hook(
                forward_layer_hook
            )
        )

    def __remove_patching_hooks(self):
        for handle in self.patching_hook_handles:
            handle.remove()
        self.patching_hook_handles = []
        self.patch_token_idx = 0

    def exploratory_patching(
        self,
        module_type,
        system_message,
        corrupt_message,
        token_idx,
        clean_token,
        corrupt_token,
        message_offset,
    ):
        # compute corrupt input and output
        corrupt_input = self.tokenizer.apply_chat_template(
            [system_message, corrupt_message], return_tensors="pt"
        ).to(self.model.device)
        corrupt_outputs = self.__generate(corrupt_input)

        metrics = []

        # begin patching process
        # for every new generated token (generated_token_idx == 0 is input sequence)
        for generated_token_idx, generated_token in enumerate(
            self.residual_cache.values()
        ):
            # skip tokens tokens after desired token (the do not have influence on the previous tokens)
            if generated_token_idx > token_idx:
                break
            # iterate through layer
            for layer_idx, layer in enumerate(generated_token.values()):
                # iterate through input tokens
                for seq_idx, seq in enumerate(layer):
                    # skip system input tokens (no need to patch these)
                    if (
                        generated_token_idx == 0
                        and seq_idx < self.system_message_length + message_offset
                    ):
                        continue
                    # register patching hook
                    if module_type == "residual":
                        self.__register_residual_patching_hook(
                            generated_token_idx, layer_idx, seq_idx
                        )
                    if module_type == "attention":
                        self.__register_attention_patching_hook(
                            generated_token_idx, layer_idx, seq_idx
                        )
                    if module_type == "mlp":
                        self.__register_mlp_patching_hook(
                            generated_token_idx, layer_idx, seq_idx
                        )
                    # generate patched output
                    patched_outputs = self.__generate(corrupt_input)
                    self.__remove_patching_hooks()

                    # corrupt output
                    corrupt_output = []
                    for token in corrupt_outputs.sequences[:, :][0]:
                        corrupt_output.append(
                            self.tokenizer.decode(token, skip_special_tokens=False)
                        )
                    # patched output
                    patched_output = []
                    for token in patched_outputs.sequences[:, :][0]:
                        patched_output.append(
                            self.tokenizer.decode(token, skip_special_tokens=False)
                        )   
                    # processed clean token
                    if len(layer) > 1:
                        processed_token_idx = seq_idx
                    else:
                        processed_token_idx = (
                            len(corrupt_input[0]) + generated_token_idx - 1
                        )
                    processed_token = self.clean_output[processed_token_idx]
                    
                    metrics.append(
                        {   
                            "layer": layer_idx,
                            "processed_token": processed_token,
                            "processed_token_id": processed_token_idx,
                            "clean_prediction": self.clean_output[
                                len(corrupt_input[0]) :
                            ],
                            "corrupt_prediction": corrupt_output[
                                len(corrupt_input[0]) :
                            ],
                            "patched_prediction": patched_output[
                                len(corrupt_input[0]) :
                            ],
                            "clean_logits": self.clean_logits[token_idx],
                            "corrupt_logits": corrupt_outputs.logits[token_idx],
                            "patched_logits": patched_outputs.logits[token_idx],
                            "clean_token": clean_token,
                            "corrupt_token": corrupt_token,
                        }
                    )
        return metrics
    
    def confirmatory_patching(
        self,
        module_type,
        system_message,
        corrupt_message,
        token_idx,
        clean_token,
        corrupt_token,
        message_offset,
        patching_token_indexs=None,
    ):
        # compute corrupt input and output
        corrupt_input = self.tokenizer.apply_chat_template(
            [system_message, corrupt_message], return_tensors="pt"
        ).to(self.model.device)
        corrupt_outputs = self.__generate(corrupt_input)

        metrics = []

        # begin patching process
        # for every new generated token (generated_token_idx == 0 is input sequence)
        for generated_token_idx, generated_token in enumerate(
            self.residual_cache.values()
        ):
            # skip tokens tokens after desired token (they do not have influence on the previous tokens)
            if generated_token_idx > token_idx:
                break
            # iterate through layer
            for layer_idx, layer in enumerate(generated_token.values()):
                # iterate through input tokens
                for seq_idx, seq in enumerate(layer):
                    # skip system input tokens (no need to patch these)
                    if (
                        generated_token_idx == 0
                        and seq_idx < self.system_message_length + message_offset
                    ):
                        continue
                    # processed clean token
                    if len(layer) > 1:
                        processed_token_idx = seq_idx
                    else:
                        processed_token_idx = (
                            len(corrupt_input[0]) + generated_token_idx - 1
                        )
                    processed_token = self.clean_output[processed_token_idx]
                    # check if current token is in the list of tokens to patch
                    if patching_token_indexs:
                        if (not generated_token_idx == token_idx) and ((processed_token_idx-self.system_message_length) not in patching_token_indexs):
                            continue
                    # register patching hook
                    if module_type == "residual":
                        self.__register_residual_patching_hook(
                            generated_token_idx, layer_idx, seq_idx
                        )
                    if module_type == "attention":
                        self.__register_attention_patching_hook(
                            generated_token_idx, layer_idx, seq_idx
                        )
                    if module_type == "mlp":
                        self.__register_mlp_patching_hook(
                            generated_token_idx, layer_idx, seq_idx
                        )
                    # generate patched output
                    patched_outputs = self.__generate(corrupt_input)
                    self.__remove_patching_hooks()

                    # corrupt output
                    corrupt_output = []
                    for token in corrupt_outputs.sequences[:, :][0]:
                        corrupt_output.append(
                            self.tokenizer.decode(token, skip_special_tokens=False)
                        )
                    # patched output
                    patched_output = []
                    for token in patched_outputs.sequences[:, :][0]:
                        patched_output.append(
                            self.tokenizer.decode(token, skip_special_tokens=False)
                        )   
                
                    metrics.append(
                        {   
                            "layer": layer_idx,
                            "processed_token": processed_token,
                            "processed_token_id": processed_token_idx,
                            "clean_prediction": self.clean_output[
                                len(corrupt_input[0]) :
                            ],
                            "corrupt_prediction": corrupt_output[
                                len(corrupt_input[0]) :
                            ],
                            "patched_prediction": patched_output[
                                len(corrupt_input[0]) :
                            ],
                            "clean_logits": self.clean_logits[token_idx],
                            "corrupt_logits": corrupt_outputs.logits[token_idx],
                            "patched_logits": patched_outputs.logits[token_idx],
                            "clean_token": clean_token,
                            "corrupt_token": corrupt_token,
                        }
                    )
        return metrics