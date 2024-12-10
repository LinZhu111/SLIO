import torch
from torch import nn



class PatchNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool
        self.coef = [0.5,1,1]
        self.num_patches = 256

    def sampler(self, feat, sample_id=None):
        B, C, H, W = feat.shape

        if sample_id is None:
            sample_id = torch.randperm(H*W, device=feat.device)
            sample_id = sample_id[:min(H*W, self.num_patches)]
            
            feat = feat.reshape(B,C,-1)
            sampled_feat = feat[:,:,sample_id].transpose(1,2)

        else:
            feat = feat.reshape(B,C,-1)
            sampled_feat = feat[:,:,sample_id].transpose(1,2)
        return sampled_feat, sample_id
    
    def forward(self, feats_spike, feats_image):
        loss = 0
        for i,(feat_spike,feat_image) in enumerate(zip(feats_spike,feats_image)):
            sampled_feat_spike, sample_id = self.sampler(feat_spike)
            sampled_feat_image, _ = self.sampler(feat_image, sample_id)

            B,P,C = sampled_feat_spike.shape
            pos = torch.bmm(sampled_feat_spike.reshape(B*P,1,C), sampled_feat_image.reshape(B*P,C,1))
            pos = pos.view(B*P, 1)

            neg = torch.bmm(sampled_feat_spike, sampled_feat_image.transpose(1,2))

            diag = torch.eye(P, device=sampled_feat_spike.device, dtype=self.mask_dtype)[None, :, :]
            neg.masked_fill_(diag, -10.0)
            neg = neg.view(B*P, P)

            out = torch.cat((pos, neg), dim=1) / 0.07

            loss =  loss+ self.coef[i]*self.loss(out, torch.zeros(B*P, dtype=torch.long,device=sampled_feat_spike.device))
            
        return loss

class PatchNCELoss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool
        self.num_patches_per_batch = 256

    def sampler(self, feat, sample_id=None):
        B, C, H, W = feat.shape

        if sample_id is None:
            sample_id = torch.randperm(H*W, device=feat.device)
            sample_id = sample_id[:min(H*W, self.num_patches_per_batch)]
            
            feat = feat.reshape(B,C,-1)
            sampled_feat = feat[:,:,sample_id].transpose(1,2).reshape(-1, C)

        else:
            feat = feat.reshape(B,C,-1)
            sampled_feat = feat[:,:,sample_id].transpose(1,2).reshape(-1, C)
        return sampled_feat, sample_id

    def forward(self, feats_q, feats_k):
        total_loss = 0
        for feat_q, feat_k in zip(feats_q, feats_k):
            feat_q, sample_id = self.sampler(feat_q)
            feat_k, _ = self.sampler(feat_k, sample_id)
            num_patches = feat_q.shape[0]
            dim = feat_q.shape[1]
            feat_k = feat_k.detach()

            # pos logit
            l_pos = torch.bmm(
                feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
            l_pos = l_pos.view(num_patches, 1)

            # neg logit

            # Should the negatives from the other samples of a minibatch be utilized?
            # In CUT and FastCUT, we found that it's best to only include negatives
            # from the same image. Therefore, we set
            # --nce_includes_all_negatives_from_minibatch as False
            # However, for single-image translation, the minibatch consists of
            # crops from the "same" high-resolution image.
            # Therefore, we will include the negatives from the entire minibatch.
                # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = num_patches // self.num_patches_per_batch

            # reshape features to batch size
            feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
            feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
            npatches = feat_q.size(1)
            l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

            # diagonal entries are similarity between same features, and hence meaningless.
            # just fill the diagonal with very small number, which is exp(-10) and almost zero
            diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
            l_neg_curbatch.masked_fill_(diagonal, -10.0)
            l_neg = l_neg_curbatch.view(-1, npatches)

            out = torch.cat((l_pos, l_neg), dim=1) / 0.07

            loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                            device=feat_q.device))
            total_loss = total_loss + loss
        return total_loss

if __name__ == '__main__':
    a = [torch.zeros(2,16,64,64), torch.zeros(2,32,32,32), torch.zeros(2,64,16,16)]
    b = [torch.zeros(2,16,64,64), torch.zeros(2,32,32,32), torch.zeros(2,64,16,16)]

    loss = PatchNCELoss2()
    l = loss(a,b)
    print()