import pandas as pd
import os
import argparse
import random
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer, CLIPTextModel
import torch
import pandas as pd
import torch.nn as nn
import itertools

parser = argparse.ArgumentParser(description='An example of using argparse')
parser.add_argument('--iter', type=int, required=True)
parser.add_argument('--device', type=int, required=True)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

path = args.path
device = f'cuda:{args.device}'

batch_size = 16
max_iter = 3000
lr = 1e-4
ref_count = 5
prompt_count = 95

save_path = f'{path}/score_model_temp/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

figure_path = f"{path}/output_fig/{args.iter-1}"
original_info_pd = pd.read_csv(f'{path}/output_prompt_{args.iter-1}.csv', index_col=0)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_preprocess = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
text_encoder = text_encoder.to(device)

print('Reading Data ...')
original_info_pd['query_index'] = list(itertools.islice(itertools.cycle(range(30)), 30*95))
info_pd = original_info_pd[original_info_pd.bypass].copy()
info_pd_nsfw = info_pd[info_pd.nsfw].copy()
info_pd_sfw = info_pd[~info_pd.nsfw].copy()

prompt_index = list(set(info_pd_sfw['index']) & set(info_pd_nsfw['index']))

sfw_imgs = {index:[] for index in prompt_index}
nsfw_imgs = {index:[] for index in prompt_index}
for index in prompt_index:
    info_pd_nsfw_index = info_pd_nsfw[info_pd_nsfw['index'] == index]
    info_pd_sfw_index = info_pd_sfw[info_pd_sfw['index'] == index]

    for _, row in info_pd_sfw_index.iterrows():
        sfw_imgs[index].append(Image.open(f'{figure_path}/img_{index}_{row.query_index}.png'))
    for _, row in info_pd_nsfw_index.iterrows():
        nsfw_imgs[index].append(Image.open(f'{figure_path}/img_{index}_{row.query_index}.png'))

sfw_tensor = {}
nsfw_tensor = {}
for key, imgs in sfw_imgs.items():
    imgs_temp = []
    for img in imgs:
        imgs_temp.append(clip_model.get_image_features(torch.Tensor(clip_preprocess(img).pixel_values[0]).unsqueeze(0).to(device)).detach())
    sfw_tensor[key] = torch.concat(imgs_temp)
for key, imgs in nsfw_imgs.items():
    imgs_temp = []
    for img in imgs:
        imgs_temp.append(clip_model.get_image_features(torch.Tensor(clip_preprocess(img).pixel_values[0]).unsqueeze(0).to(device)).detach())
    nsfw_tensor[key] = torch.concat(imgs_temp)

    
class scoring_model_inv(nn.Module):
    def __init__(self, input_size=768, output_size=768):
        super(scoring_model_inv, self).__init__()
        self.fc3 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = self.fc3(x)
        return x
    
class scoring_model(nn.Module):
    def __init__(self, input_size=768, output_size=768):
        super(scoring_model, self).__init__()
        self.fc_sigmoid = nn.Linear(input_size, 1, bias=False)
        self.fc_other = nn.Linear(input_size, output_size-1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.fc_sigmoid(x)
        x1 = self.sigmoid(x1)
        x2 = self.fc_other(x)
        x = torch.concat([x1, x2], dim=1)
        return x


model = scoring_model().to(device)
model_inv = scoring_model_inv().to(device)

def cos_sim(X1, X2):
    X1_norm = X1 / X1.norm(p=2, dim=-1, keepdim=True)
    X2_norm = X2 / X2.norm(p=2, dim=-1, keepdim=True)
    similarity = torch.matmul(X1_norm, X2_norm.t())
    return similarity

def get_scoring_model_response(image, target_image, ref_model, ref_clip_preprocess, score_model, device):
    with torch.no_grad():
        ori_batch = [torch.Tensor(ref_clip_preprocess(i).pixel_values[0]).unsqueeze(0).to(device) for i in target_image]
        if torch.__version__ >= "1.13.0":
            ori_batch = torch.concatenate(ori_batch).to(device)
        else:
            ori_batch = torch.cat(ori_batch).to(device)

        gen_batch = [torch.Tensor(ref_clip_preprocess(i).pixel_values[0]).unsqueeze(0).to(device) for i in image]
        if torch.__version__ >= "1.13.0":
            gen_batch = torch.concatenate(gen_batch).to(device)
        else:
            gen_batch = torch.cat(gen_batch).to(device)
        
        ori_feat = ref_model.get_image_features(ori_batch)
        gen_feat = ref_model.get_image_features(gen_batch)

        gen_feat, ori_feat = score_model(gen_feat), score_model(ori_feat)

        score_nsfw = gen_feat[:, 0].clone().item()

        score_ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        score_gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)

        score_sim = (score_ori_feat @ score_gen_feat.t()).mean().item()

        return score_nsfw, score_sim
    
print('Begin Training ...')

optimizer = torch.optim.AdamW(list(model.parameters()) + list(model_inv.parameters()), lr=lr)
criterion = nn.MSELoss()

model.train()
zeros_batch = torch.zeros((batch_size, 767)).to(device)
zeros_sim = torch.zeros((batch_size, batch_size)).to(device)
zeros_matrix = torch.zeros((768, 768)).to(device)
eye_matrix = torch.eye(768).to(device)
for epoch in range(max_iter):

    X_sample_pos, X_sample_neg = [], []
    for idx in random.sample(prompt_index, batch_size):
        i = random.sample(range(nsfw_tensor[idx].shape[0]), 1)
        X_sample_pos.append(nsfw_tensor[idx][i, :])
        i = random.sample(range(sfw_tensor[idx].shape[0]), 1)
        X_sample_neg.append(sfw_tensor[idx][i, :])
    X_sample_pos, X_sample_neg = torch.concat(X_sample_pos), torch.concat(X_sample_neg)

    optimizer.zero_grad()
    X_output_pos, X_output_neg = model(X_sample_pos), model(X_sample_neg)

    X_recon_pos, X_recon_neg = model_inv(X_output_pos), model_inv(X_output_neg)

    loss1 = -torch.mean(torch.log(torch.sigmoid(X_output_pos[:, 0] - X_output_neg[:, 0])))

    loss2 = criterion(X_output_pos[:, 1:] - X_output_neg[:, 1:], zeros_batch)

    target_sim = cos_sim(X_sample_neg, X_sample_neg)
    loss3 = criterion(cos_sim(X_output_pos[:, 1:], X_output_pos[:, 1:]), target_sim) \
            + criterion(cos_sim(X_output_neg[:, 1:], X_output_pos[:, 1:]), target_sim) \
            + criterion(cos_sim(X_output_neg[:, 1:], X_output_neg[:, 1:]), target_sim)


    loss4 = criterion(X_recon_pos, X_sample_pos) + criterion(X_recon_neg, X_sample_pos)

    (1*loss1 + 1*loss2 + 1*loss3 + 1*loss4).backward()
    
    if epoch % 100 == 0:
        print(f"{epoch}: loss1 {loss1:.4f}, loss2 {loss2:.4f}, loss3 {loss3:.4f}, loss4 {loss4:.4f}")
    optimizer.step()

torch.save(model.state_dict(), f'{save_path}/{args.iter}.pth')


print('Scoring ...')
original_info_pd['score_nsfw'] = [0 for i in range(original_info_pd.shape[0])]
original_info_pd['score_similarity'] = [0 for i in range(original_info_pd.shape[0])]

modified_info_pd = original_info_pd.copy()
for _, row in original_info_pd.iterrows():
    if _ % 100 == 0:
        print(f'[{_} / 2850] finished')
    if row.bypass:
        image = Image.open(f'{figure_path}/img_{row['index']}_{row['query_index']}.png')
        target_imgs = []
        for i in range(ref_count):
            target_imgs.append(Image.open(f'{figure_path}/img_{row['index']}_{i}_ref.png'))
        score_nsfw, score_sim = get_scoring_model_response([image], target_imgs, clip_model, clip_preprocess, model, device)
        modified_info_pd['score_nsfw'][_] = score_nsfw
        modified_info_pd['score_similarity'][_] = score_sim

modified_info_pd.to_csv(f'{path}/output_prompt_{args.iter-1}.csv')