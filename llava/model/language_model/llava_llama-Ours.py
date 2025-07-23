#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from time import time
import json
import pickle
import os
def max_sqrt_sum_with_indices(l1, l2, K):
    device = l1.device
    l1 = torch.tensor(l1, dtype=torch.float32)
    l2 = torch.tensor(l2, dtype=torch.float32)
    
    # Sort in descending order and keep track of original indices
    l1_sorted, l1_indices = torch.sort(l1, descending=True)
    l2_sorted, l2_indices = torch.sort(l2, descending=True)
    
    # Compute prefix sums
    l1_prefix_sum = torch.cat((torch.tensor([0.0], device=device), torch.cumsum(l1_sorted, dim=0)))
    l2_prefix_sum = torch.cat((torch.tensor([0.0], device=device), torch.cumsum(l2_sorted, dim=0)))
    
    # Create index tensors
    k1_values = torch.arange(min(K, len(l1)) + 1)
    k2_values = K - k1_values
    
    # Mask out invalid k2 values
    valid_mask = (k2_values >= 0) & (k2_values <= len(l2))
    
    # Select valid prefix sums
    valid_k1_sums = l1_prefix_sum[k1_values[valid_mask]]
    valid_k2_sums = l2_prefix_sum[k2_values[valid_mask]]
    
    # Compute potential values
    potential_values = valid_k1_sums * valid_k2_sums
    
    # Get the index of the maximum value
    max_index = torch.argmax(potential_values)
    
    # Get the corresponding k1 and k2
    best_k1 = k1_values[valid_mask][max_index].item()
    best_k2 = k2_values[valid_mask][max_index].item()
    
    # Get the indices of the selected elements
    selected_l1_indices = l1_indices[:best_k1].tolist()
    selected_l2_indices = l2_indices[:best_k2].tolist()
    
    return selected_l1_indices, selected_l2_indices, torch.sqrt(potential_values[max_index])


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, visual_token_num=None):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # FasterVLM
        self.visual_token_num = visual_token_num

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # FasterVLM
    def get_visual_token_num(self):
        return self.visual_token_num

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                v_token_num,
                cls_attn
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            v_token_num, cls_attn = 0, None

        if cls_attn == None:
            return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        ), v_token_num, cls_attn
        """
        The following lines are modified by @jiayi han for token selection
        """

        # ====================================================
        #      modification version 1: T2I attn. only
        # ====================================================
        """
        N_img = cls_attn.shape[0]
        base = 144
        k = int(base / N_img)

        v_token_num = k * N_img # whole budget
        B, N, C = inputs_embeds.shape
        forward_tokens = inputs_embeds[:, 35:, :]
        pre_tokens = inputs_embeds[:,:35,:]
        image_tokens = forward_tokens[:, :(576*N_img), :]
        text_tokens = forward_tokens[:, (576*N_img):, :]
    
        simi = torch.matmul(F.normalize(text_tokens, p=2, dim=-1), 
                            F.normalize(image_tokens, p=2, dim=-1).permute(0,2,1))
        simi = (simi*100).softmax(-1).max(1)[0] # B, Ni
        # for the T2I attn.
        value, indices = torch.topk(simi, k=(k*N_img), dim=-1) # B, k
        indices = torch.sort(indices, dim=1)[0] # (B, T)
        
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, (k*N_img))
        image_tokens = image_tokens[batch_indices, indices, ...]
        inputs_embeds = torch.cat([pre_tokens, image_tokens, text_tokens], dim=1)
        """
        # ====================================================
        #    modification version 2: T2I attn. + [CLS] attn.
        # ====================================================
        """
        N_img = cls_attn.shape[0]
        base = 360
        k = int(base / N_img)
        B, N, C = inputs_embeds.shape
        forward_tokens = inputs_embeds[:, 35:, :]
        pre_tokens = inputs_embeds[:,:35,:]
        image_tokens = forward_tokens[:, :(576*N_img), :]
        text_tokens = forward_tokens[:, (576*N_img):, :]

        simi = torch.matmul(F.normalize(text_tokens, p=2, dim=-1), 
                            F.normalize(image_tokens, p=2, dim=-1).permute(0,2,1)).max(1)[0] # B, Ni
        
        # for the T2I attn.
        _, indices = torch.topk(simi, k=(k*N_img), dim=-1) # B, k
        # for the [CLS] attn.
        _, indices_CLS = torch.topk(cls_attn, k=k, dim=-1) # B, k
        
        base_indices = torch.ones_like(indices_CLS,device=indices_CLS.device) * torch.arange(N_img, device=indices_CLS.device)[:, None]
        indices_CLS = indices_CLS + base_indices
        indices_CLS = indices_CLS.view(1, -1)
        
        indices = torch.cat([indices, indices_CLS], -1) # B, 2k
        indices = torch.sort(indices, dim=1)[0] # (B, T)
        
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, 2*(k*N_img))
        image_tokens = image_tokens[batch_indices, indices, ...]
        inputs_embeds = torch.cat([pre_tokens, image_tokens, text_tokens], dim=1)

        v_token_num = k * 2 * N_img
        """
        # ====================================================
        #  modification version 3: Mix T2I attn. + [CLS] attn.
        # ====================================================
        
        N_img = cls_attn.shape[0]
        base = 29
        k = int(base / N_img)

        v_token_num = k * 2 * N_img # whole budget
        
        budget_thres = 0.25
        B, N, C = inputs_embeds.shape
        forward_tokens = inputs_embeds[:, 35:, :]
        pre_tokens = inputs_embeds[:,:35,:]
        image_tokens = forward_tokens[:, :(576*N_img), :]
        text_tokens = forward_tokens[:, (576*N_img):, :]
    
        simi = torch.matmul(F.normalize(text_tokens, p=2, dim=-1), 
                            F.normalize(image_tokens, p=2, dim=-1).permute(0,2,1))
        simi = (simi*100).softmax(-1).max(1)[0] # B, Ni
        
        # for the T2I attn.
        value, indices = torch.topk(simi, k=(k*N_img*2), dim=-1) # B, 2k
        # for the [CLS] attn.
        value_CLS, indices_CLS = torch.topk(cls_attn, k=k*2, dim=-1) # B, 2k
    
        value_CLS = value_CLS.reshape(1, -1)
        #value = value / value.sum(-1, keepdim=True)
        #value_CLS = value_CLS / value_CLS.sum(-1, keepdim=True)
    
        base_indices = torch.ones_like(indices_CLS,device=indices_CLS.device) * torch.arange(N_img, device=indices_CLS.device)[:, None]
        indices_CLS = indices_CLS + base_indices
        indices_CLS = indices_CLS.view(1, -1)
        simi_indices = []
        CLS_indices = []

        for v, vc in zip(value, value_CLS):
            base_si = list(range(int(2*(k*N_img)*budget_thres)))
            base_ci = list(range(int(2*(k*N_img)*budget_thres)))
            si, ci, _ = max_sqrt_sum_with_indices(v[len(base_si):], vc[len(base_ci):], v_token_num-2*len(base_ci))
            si = base_si+[s+len(base_si) for s in si]
            ci = base_ci+[s+len(base_ci) for s in ci]

            simi_indices.append(si)
            CLS_indices.append(ci)

        MI = []
        for i in range(len(indices_CLS)):
            MI.append(torch.cat([indices[i,simi_indices[i]], indices_CLS[i,CLS_indices[i]]]))
        indices = torch.stack(MI, 0)
    
        #indices = indices[torch.arange(B).unsqueeze(1).expand(-1, 2*(k*N_img)), mixed_indices]
        indices = torch.sort(indices, dim=1)[0] # (B, T)
        
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, 2*(k*N_img))
        image_tokens = image_tokens[batch_indices, indices, ...]
        inputs_embeds = torch.cat([pre_tokens, image_tokens, text_tokens], dim=1)
        
        # =====================================================
        #   the following lines are solely for bias analysis
        # =====================================================
        """
        D = dict(value=value.detach().cpu().numpy().tolist(), value_CLS=value_CLS.detach().cpu().numpy().tolist(), N_img=N_img, budget_thres=budget_thres)
        S = json.dumps(D)
        with open("/data/user/hanjy/FasterVLM/playground/data/distribution/analyze/gqa/dist.jsonl","a") as f:
            f.writelines(S+"\n")
            f.close()
        """
        """
        N_img = cls_attn.shape[0]
        B, N, C = inputs_embeds.shape
        forward_tokens = inputs_embeds[:, 35:, :]
        pre_tokens = inputs_embeds[:,:35,:]
        image_tokens = forward_tokens[:, :(576*N_img), :]
        text_tokens = forward_tokens[:, (576*N_img):, :]
    
        simi = torch.matmul(F.normalize(text_tokens, p=2, dim=-1), 
                            F.normalize(image_tokens, p=2, dim=-1).permute(0,2,1))
        #simi = torch.matmul(text_tokens, 
        #                    image_tokens.permute(0,2,1))
        simi = (simi*100).softmax(-1).max(1)[0] # B, Ni
        D = dict(value=simi.detach().cpu().numpy().tolist())
        S = json.dumps(D)
        with open("/data/user/hanjy/FasterVLM/playground/data/distribution/analyze/simi.jsonl","a") as f:
            f.writelines(S+"\n")
            f.close()
        
        """
        dataset = "pope"
        mask = torch.zeros(576).float()
        mask[indices]=1.
        mask = mask.reshape(24,24)
        imgs = next(os.walk(f"/data/user/hanjy/FasterVLM/playground/data/analysis/{dataset}/img_attn"))[1]
        imgs = [i for i in imgs if "." not in i]
            
        if len(imgs)==0:
            idx = 0
        else:
            idx = max([int(i) for i in imgs if "." not in i])

        mask = torch.kron(mask,torch.ones(14,14,device=mask.device))
        os.makedirs(f"/data/user/hanjy/FasterVLM/playground/data/analysis/{dataset}/img_attn/{idx+1}",exist_ok=True)
        with open(f"/data/user/hanjy/FasterVLM/playground/data/analysis/{dataset}/img_attn/{idx+1}/hmap.pkl","wb") as f:
            pickle.dump(mask,f)
            f.close()
        #plt.savefig(f"/data/user/hanjy/FasterVLM/playground/data/analysis/pope/img_attn/{idx}/hmap.png", bbox_inches='tight')
        
        # =====================================================
        #                      final output
        # =====================================================
        
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        ), v_token_num, cls_attn

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
