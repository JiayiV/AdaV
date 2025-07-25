import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    # FasterVLM
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_attentions = image_forward_outs.attentions[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
            image_attentions = image_attentions[:, :, 0, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
            image_attentions = image_attentions
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features, image_attentions

    # FasterVLM
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features, image_attentions = [], []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_attentions=True, output_hidden_states=True)
                image_feature, image_attention = self.feature_select(image_forward_out)
                image_features.append(image_feature.to(image.dtype))
                image_attentions.append(image_attention.to(image.dtype))
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_attentions=True, output_hidden_states=True)
            image_features, image_attentions = self.feature_select(image_forward_outs)
            image_features, image_attentions = image_features.to(images.dtype), image_attentions.to(images.dtype)
        """
        print("llava/model/multimodel_encoder/clip_encoder.py, line 63")
        print("this is the notation for file saving, eliminate for normal mode")
        import pdb
        pdb.set_trace()
        # get the image order
        dirs = next(os.walk("/data/user/hanjy/FasterVLM/playground/data/analysis/img_attn"))[1]
        path = f"/data/user/hanjy/FasterVLM/playground/data/analysis/img_attn/{len(dirs)-2}"
        input_tensor = image_forward_outs["hidden_states"][0][0,:-1,:]
        .mean(-1).reshape(24,24)[3:21,:]
        output_tensor = F.interpolate(input_tensor, size=(480, 640), mode='bilinear', align_corners=False)

        plt.imshow(output_tensor.detach().cpu().numpy())
        plt.axis('off')
        
        plt.savefig(os.path.join(path, f"feature_embedding.png"))
        plt.close()
        
        for i in range(24):
            attn = image_forward_outs['attentions'][i]
            all_heads = attn[0,...].sum(0)
            input_tensor = all_heads[276,1:].mean(-1).reshape(24,24)[3:21,:]
            output_tensor = F.interpolate(input_tensor, size=(480, 640), mode='bilinear', align_corners=False)
            plt.imshow(output_tensor.detach().cpu().numpy(), cmap='viridis')
            plt.axis('off')
            
            plt.title(f'all_heads Attention Map - Layer {i}', fontsize=16)
            plt.savefig(os.path.join(path, f"all_heads_{i}.png"))
            plt.close()

            input_tensor = image_forward_outs["hidden_states"][i][0,:-1,:].mean(-1).reshape(24,24)[3:21,:]
            output_tensor = F.interpolate(input_tensor, size=(480, 640), mode='bilinear', align_corners=False)
            
            plt.imshow(output_tensor.detach().cpu().numpy())
            plt.axis('off')
            
            plt.savefig(os.path.join(path, f"feature_{i}.png"))
            plt.close()
            
            for j in range(attn.shape[1]):
                this_head = attn[0,j,...]
                input_tensor = image_forward_outs["hidden_states"][i][0,:-1,:].mean(-1).reshape(24,24)[3:21,:]
                output_tensor = F.interpolate(input_tensor, size=(480, 640), mode='bilinear', align_corners=False)
                plt.imshow(output_tensor.detach().cpu().numpy(), cmap='viridis')
                plt.axis('off')
            
                plt.title(f'head Attention Map - Layer {i} Head {j}', fontsize=16)
                plt.savefig(os.path.join(path, f"layer_{i}_head_{j}.png"))
                plt.close()
        """
        return image_features, image_attentions

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
