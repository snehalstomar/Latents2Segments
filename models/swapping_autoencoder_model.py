import torch
import util
from models import BaseModel
import models.networks as networks
import models.networks.loss as loss


class SwappingAutoencoderModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        BaseModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--spatial_code_ch", default=8, type=int)
        parser.add_argument("--global_code_ch", default=2048, type=int)
        parser.add_argument("--lambda_R1", default=10.0, type=float)
        parser.add_argument("--lambda_patch_R1", default=1.0, type=float)
        parser.add_argument("--lambda_L1", default=1.0, type=float)
        parser.add_argument("--lambda_GAN", default=1.0, type=float)
        parser.add_argument("--lambda_PatchGAN", default=1.0, type=float)
        parser.add_argument("--patch_min_scale", default=1 / 8, type=float)
        parser.add_argument("--patch_max_scale", default=1 / 4, type=float)
        parser.add_argument("--patch_num_crops", default=8, type=int)
        parser.add_argument("--patch_use_aggregation",
                            type=util.str2bool, default=True)
        return parser

    def initialize(self):
        self.E = networks.create_network(self.opt, self.opt.netE, "encoder")
        self.G = networks.create_network(self.opt, self.opt.netG, "generator")
        if self.opt.lambda_GAN > 0.0:
            self.D = networks.create_network(
                self.opt, self.opt.netD, "discriminator")
        if self.opt.lambda_PatchGAN > 0.0:
            self.Dpatch = networks.create_network(
                self.opt, self.opt.netPatchD, "patch_discriminator"
            )

        # Count the iteration count of the discriminator
        # Used for lazy R1 regularization (c.f. Appendix B of StyleGAN2)
        self.register_buffer(
            "num_discriminator_iters", torch.zeros(1, dtype=torch.long)
        )
        self.l1_loss = torch.nn.L1Loss()

        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load()

        if self.opt.num_gpus > 0:
            self.to("cuda:0")
        self.mask_struct_1, self.mask_struct_2, self.mask_struct_3, self.mask_struct_4, self.mask_struct_5 = None, None, None, None, None    
        
    
    def mask_maker(self):
        # mask_struct_1, mask_struct_2, mask_struct_3, mask_struct_4, mask_struct_5 = torch.zeros(4, 8, 8, 8), torch.zeros(4, 8, 8, 8), torch.zeros(4, 8, 8, 8), torch.zeros(4, 8, 8, 8), torch.zeros(4, 8, 8, 8)
        mask_struct_1, mask_struct_2, mask_struct_3, mask_struct_4, mask_struct_5 = torch.zeros(2, 8, 8, 8), torch.zeros(2, 8, 8, 8), torch.zeros(2, 8, 8, 8), torch.zeros(2, 8, 8, 8), torch.zeros(2, 8, 8, 8)        
        #mask_texture_1, mask_texture_2 = torch.zeros(2, 2048), torch.zeros(2, 2048)
        #ones_mask_struct = torch.ones(2, 1, 8, 8)
        # ones_mask_struct = torch.ones(4, 2, 8, 8)
        # ones_mask_struct_1 = torch.ones(4, 1, 8, 8)
        ones_mask_struct = torch.ones(2, 2, 8, 8)
        ones_mask_struct_1 = torch.ones(2, 1, 8, 8)
        #ones_mask_texture = torch.ones(2, 1024)     
        mask_struct_1[:, 0: 2, :, :] = ones_mask_struct
        mask_struct_2[:, 2: 4, :, :] = ones_mask_struct
        mask_struct_3[:, 4: 6, :, :] = ones_mask_struct
        mask_struct_4[:, 6: 7, :, :] = ones_mask_struct_1
        mask_struct_5[:, 7: 8, :, :] = ones_mask_struct_1
        # mask_texture_1[:, 0: 1024] = ones_mask_texture
        # mask_texture_2[:, 1024: 2048] = ones_mask_texture
        return mask_struct_1, mask_struct_2, mask_struct_3, mask_struct_4, mask_struct_5

    def per_gpu_initialize(self):
        pass

    def swap(self, x):
        """ Swaps (or mixes) the ordering of the minibatch """
        shape = x.shape
        assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
        new_shape = [shape[0] // 2, 2] + list(shape[1:])
        x = x.view(*new_shape)
        x = torch.flip(x, [1])
        return x.view(*shape)

    def compute_image_discriminator_losses(self, real, rec, mix):
        if self.opt.lambda_GAN == 0.0:
            return {}

        pred_real = self.D(real)
        pred_rec = self.D(rec)
        pred_mix = self.D(mix)

        losses = {}
        losses["D_real"] = loss.gan_loss(
            pred_real, should_be_classified_as_real=True
        ) * self.opt.lambda_GAN

        losses["D_rec"] = loss.gan_loss(
            pred_rec, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)
        losses["D_mix"] = loss.gan_loss(
            pred_mix, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)

        return losses

    def get_random_crops(self, x, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.
        """
        crops = util.apply_random_crop(
            x, self.opt.patch_size,
            (self.opt.patch_min_scale, self.opt.patch_max_scale),
            num_crops=self.opt.patch_num_crops
        )
        return crops

    def compute_patch_discriminator_losses(self, real, mix):
        losses = {}
        real_feat = self.Dpatch.extract_features(
            self.get_random_crops(real),
            aggregate=self.opt.patch_use_aggregation
        )
        target_feat = self.Dpatch.extract_features(self.get_random_crops(real))
        mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))

        losses["PatchD_real"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, target_feat),
            should_be_classified_as_real=True,
        ) * self.opt.lambda_PatchGAN

        losses["PatchD_mix"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, mix_feat),
            should_be_classified_as_real=False,
        ) * self.opt.lambda_PatchGAN

        return losses

    def compute_discriminator_losses(self, real, images_feat_1, images_feat_2, images_feat_3, images_feat_4, images_feat_5):
        self.num_discriminator_iters.add_(1)
        B = real.size(0)
        #print('B->', B)
        assert B % 2 == 0, "Batch size must be even on each GPU."
        
        #print(sp.shape, gl.shape)
        mask_struct_1, mask_struct_2, mask_struct_3, mask_struct_4, mask_struct_5 = self.mask_maker()
        #print(sp.shape)
        sp, gl = self.E(real)
        #print('sp.shape->', sp.shape, 'gl.shape->', gl.shape)
        sp_feat_1 = torch.mul(sp, mask_struct_1.to('cuda'))
        sp_feat_2 = torch.mul(sp, mask_struct_2.to('cuda'))
        sp_feat_3 = torch.mul(sp, mask_struct_3.to('cuda'))
        sp_feat_4 = torch.mul(sp, mask_struct_4.to('cuda'))
        sp_feat_5 = torch.mul(sp, mask_struct_5.to('cuda'))

        # To save memory, compute the GAN loss on only
        # half of the reconstructed images
        # rec = self.G(sp[:B // 2], gl[:B // 2])
        rec_feat_1 = self.G(sp_feat_1[:B // 2], gl[:B // 2])        
        rec_feat_2 = self.G(sp_feat_2[:B // 2], gl[:B // 2])
        rec_feat_3 = self.G(sp_feat_3[:B // 2], gl[:B // 2])        
        rec_feat_4 = self.G(sp_feat_4[:B // 2], gl[:B // 2])
        rec_feat_5 = self.G(sp_feat_5[:B // 2], gl[:B // 2])        

        # mix = self.G(self.swap(sp), gl)
        mix_feat_1 = self.G(self.swap(sp_feat_1), gl)
        mix_feat_2 = self.G(self.swap(sp_feat_2), gl)
        mix_feat_3 = self.G(self.swap(sp_feat_3), gl)
        mix_feat_4 = self.G(self.swap(sp_feat_4), gl)
        mix_feat_5 = self.G(self.swap(sp_feat_5), gl)

        losses = {}
        #losses_full = self.compute_image_discriminator_losses(real, rec, mix)
        losses_feat_1 = self.compute_image_discriminator_losses(images_feat_1, rec_feat_1, mix_feat_1)
        losses_feat_2 = self.compute_image_discriminator_losses(images_feat_2, rec_feat_2, mix_feat_2)
        losses_feat_3 = self.compute_image_discriminator_losses(images_feat_3, rec_feat_3, mix_feat_3)
        losses_feat_4 = self.compute_image_discriminator_losses(images_feat_4, rec_feat_4, mix_feat_4)
        losses_feat_5 = self.compute_image_discriminator_losses(images_feat_5, rec_feat_5, mix_feat_5)

        losses['D_real'] = (0.2 * losses_feat_1['D_real']) + (0.2 * losses_feat_2['D_real']) + (0.2 * losses_feat_3['D_real']) + (0.2 * losses_feat_4['D_real']) + (0.2 * losses_feat_5['D_real']) 
        losses['D_rec'] =  (0.2 * losses_feat_1['D_rec']) + (0.2 * losses_feat_2['D_rec']) + (0.2 * losses_feat_3['D_rec']) + (0.2 * losses_feat_4['D_rec']) + (0.2 * losses_feat_5['D_rec']) 
        losses['D_mix'] =  (0.2 * losses_feat_1['D_mix']) + (0.2 * losses_feat_2['D_mix']) + (0.2 * losses_feat_3['D_mix']) + (0.2 * losses_feat_4['D_mix']) + (0.2 * losses_feat_5['D_mix'])         
        if self.opt.lambda_PatchGAN > 0.0:
            #patch_losses_full = self.compute_patch_discriminator_losses(real, mix)
            patch_losses_feat_1 = self.compute_patch_discriminator_losses(images_feat_1, mix_feat_1)
            patch_losses_feat_2 = self.compute_patch_discriminator_losses(images_feat_2, mix_feat_2)
            patch_losses_feat_3 = self.compute_patch_discriminator_losses(images_feat_3, mix_feat_3)
            patch_losses_feat_4 = self.compute_patch_discriminator_losses(images_feat_4, mix_feat_4)
            patch_losses_feat_5 = self.compute_patch_discriminator_losses(images_feat_5, mix_feat_5)
            
            losses["PatchD_real"] = (0.2 * patch_losses_feat_1["PatchD_real"]) + (0.2 * patch_losses_feat_2["PatchD_real"]) + (0.2 * patch_losses_feat_3["PatchD_real"]) + (0.2 * patch_losses_feat_4["PatchD_real"]) + (0.2 * patch_losses_feat_5["PatchD_real"])     
            losses["PatchD_mix"] = (0.2 * patch_losses_feat_1["PatchD_mix"]) + (0.2 * patch_losses_feat_2["PatchD_mix"]) + (0.2 * patch_losses_feat_3["PatchD_mix"]) + (0.2 * patch_losses_feat_4["PatchD_mix"]) + (0.2 * patch_losses_feat_5["PatchD_mix"])                    
            
        metrics = {}  # no metrics to report for the Discriminator iteration
        return losses, metrics, sp.detach(), gl.detach()

    def compute_R1_loss(self, real, real_feat_1, real_feat_2, real_feat_3, real_feat_4, real_feat_5):
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            # real.requires_grad_()
            real_feat_1.requires_grad_()
            real_feat_2.requires_grad_()
            real_feat_3.requires_grad_()
            real_feat_4.requires_grad_()
            real_feat_5.requires_grad_()

            # pred_real = self.D(real).sum()
            pred_real_feat_1 = self.D(real_feat_1).sum()
            pred_real_feat_2 = self.D(real_feat_2).sum()
            pred_real_feat_3 = self.D(real_feat_3).sum()
            pred_real_feat_4 = self.D(real_feat_4).sum()
            pred_real_feat_5 = self.D(real_feat_5).sum()

            # grad_real, = torch.autograd.grad(outputs=pred_real, inputs=[real], create_graph=True, retain_graph=True,)
            grad_real_feat_1, = torch.autograd.grad(outputs=pred_real_feat_1, inputs=[real_feat_1], create_graph=True, retain_graph=True,)
            grad_real_feat_2, = torch.autograd.grad(outputs=pred_real_feat_2, inputs=[real_feat_2], create_graph=True, retain_graph=True,)
            grad_real_feat_3, = torch.autograd.grad(outputs=pred_real_feat_3, inputs=[real_feat_3], create_graph=True, retain_graph=True,)
            grad_real_feat_4, = torch.autograd.grad(outputs=pred_real_feat_4, inputs=[real_feat_4], create_graph=True, retain_graph=True,)
            grad_real_feat_5, = torch.autograd.grad(outputs=pred_real_feat_5, inputs=[real_feat_5], create_graph=True, retain_graph=True,)
            
            # grad_real2 = grad_real.pow(2)
            grad_real2_feat_1 = grad_real_feat_1.pow(2)
            grad_real2_feat_2 = grad_real_feat_2.pow(2)
            grad_real2_feat_3 = grad_real_feat_3.pow(2)
            grad_real2_feat_4 = grad_real_feat_4.pow(2)
            grad_real2_feat_5 = grad_real_feat_5.pow(2)
    
            # dims = list(range(1, grad_real2.ndim))
            dims_feat_1 = list(range(1, grad_real2_feat_1.ndim))
            dims_feat_2 = list(range(1, grad_real2_feat_2.ndim))
            dims_feat_3 = list(range(1, grad_real2_feat_3.ndim))
            dims_feat_4 = list(range(1, grad_real2_feat_4.ndim))
            dims_feat_5 = list(range(1, grad_real2_feat_5.ndim))
            
            # grad_penalty_full = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
            grad_penalty_feat_1 = grad_real2_feat_1.sum(dims_feat_1) * (self.opt.lambda_R1 * 0.5)
            grad_penalty_feat_2 = grad_real2_feat_2.sum(dims_feat_2) * (self.opt.lambda_R1 * 0.5)
            grad_penalty_feat_3 = grad_real2_feat_3.sum(dims_feat_3) * (self.opt.lambda_R1 * 0.5)
            grad_penalty_feat_4 = grad_real2_feat_4.sum(dims_feat_4) * (self.opt.lambda_R1 * 0.5)
            grad_penalty_feat_5 = grad_real2_feat_5.sum(dims_feat_5) * (self.opt.lambda_R1 * 0.5)

            grad_penalty = (0.2 * grad_penalty_feat_1) + (0.2 * grad_penalty_feat_2) + (0.2 * grad_penalty_feat_3) + (0.2 * grad_penalty_feat_4) + (0.2 * grad_penalty_feat_5) 
        else:
            grad_penalty = 0.0

        if self.opt.lambda_patch_R1 > 0.0:
            # real_crop = self.get_random_crops(real).detach()
            # real_crop.requires_grad_()
            # target_crop = self.get_random_crops(real).detach()
            # target_crop.requires_grad_()

            real_crop_feat_1 = self.get_random_crops(real_feat_1).detach()
            real_crop_feat_1.requires_grad_()
            target_crop_feat_1 = self.get_random_crops(real_feat_1).detach()
            target_crop_feat_1.requires_grad_()

            real_crop_feat_2 = self.get_random_crops(real_feat_2).detach()
            real_crop_feat_2.requires_grad_()
            target_crop_feat_2 = self.get_random_crops(real_feat_2).detach()
            target_crop_feat_2.requires_grad_()

            real_crop_feat_3 = self.get_random_crops(real_feat_3).detach()
            real_crop_feat_3.requires_grad_()
            target_crop_feat_3 = self.get_random_crops(real_feat_3).detach()
            target_crop_feat_3.requires_grad_()

            real_crop_feat_4 = self.get_random_crops(real_feat_4).detach()
            real_crop_feat_4.requires_grad_()
            target_crop_feat_4 = self.get_random_crops(real_feat_4).detach()
            target_crop_feat_4.requires_grad_()

            real_crop_feat_5 = self.get_random_crops(real_feat_5).detach()
            real_crop_feat_5.requires_grad_()
            target_crop_feat_5 = self.get_random_crops(real_feat_5).detach()
            target_crop_feat_5.requires_grad_()

            # real_feat_full = self.Dpatch.extract_features(real_crop, aggregate=self.opt.patch_use_aggregation)
            # target_feat_full = self.Dpatch.extract_features(target_crop)
            # pred_real_patch_full = self.Dpatch.discriminate_features(real_feat_full, target_feat_full).sum()

            real_feat_feat_1 = self.Dpatch.extract_features(real_crop_feat_1, aggregate=self.opt.patch_use_aggregation)
            target_feat_feat_1 = self.Dpatch.extract_features(target_crop_feat_1)
            pred_real_patch_feat_1 = self.Dpatch.discriminate_features(real_feat_feat_1, target_feat_feat_1).sum()

            real_feat_feat_2 = self.Dpatch.extract_features(real_crop_feat_2, aggregate=self.opt.patch_use_aggregation)
            target_feat_feat_2 = self.Dpatch.extract_features(target_crop_feat_2)
            pred_real_patch_feat_2 = self.Dpatch.discriminate_features(real_feat_feat_2, target_feat_feat_2).sum()
            
            real_feat_feat_3 = self.Dpatch.extract_features(real_crop_feat_3, aggregate=self.opt.patch_use_aggregation)
            target_feat_feat_3 = self.Dpatch.extract_features(target_crop_feat_3)
            pred_real_patch_feat_3 = self.Dpatch.discriminate_features(real_feat_feat_3, target_feat_feat_3).sum()

            real_feat_feat_4 = self.Dpatch.extract_features(real_crop_feat_4, aggregate=self.opt.patch_use_aggregation)
            target_feat_feat_4 = self.Dpatch.extract_features(target_crop_feat_4)
            pred_real_patch_feat_4 = self.Dpatch.discriminate_features(real_feat_feat_4, target_feat_feat_4).sum()

            real_feat_feat_5 = self.Dpatch.extract_features(real_crop_feat_5, aggregate=self.opt.patch_use_aggregation)
            target_feat_feat_5 = self.Dpatch.extract_features(target_crop_feat_5)
            pred_real_patch_feat_5 = self.Dpatch.discriminate_features(real_feat_feat_5, target_feat_feat_5).sum()
            

            # grad_real_full, grad_target_full = torch.autograd.grad(outputs = pred_real_patch_full, inputs = [real_crop, target_crop], 
            #                                    create_graph=True, retain_graph=True,)
            grad_real_feat_1, grad_target_feat_1 = torch.autograd.grad(outputs = pred_real_patch_feat_1, inputs=[real_crop_feat_1, target_crop_feat_1], 
                                               create_graph=True, retain_graph=True,)
            grad_real_feat_2, grad_target_feat_2 = torch.autograd.grad(outputs = pred_real_patch_feat_2, inputs=[real_crop_feat_2, target_crop_feat_2], 
                                               create_graph=True, retain_graph=True,)
            grad_real_feat_3, grad_target_feat_3 = torch.autograd.grad(outputs = pred_real_patch_feat_3, inputs=[real_crop_feat_3, target_crop_feat_3], 
                                               create_graph=True, retain_graph=True,)
            grad_real_feat_4, grad_target_feat_4 = torch.autograd.grad(outputs = pred_real_patch_feat_4, inputs=[real_crop_feat_4, target_crop_feat_4], 
                                               create_graph=True, retain_graph=True,)
            grad_real_feat_5, grad_target_feat_5 = torch.autograd.grad(outputs = pred_real_patch_feat_5, inputs=[real_crop_feat_5, target_crop_feat_5], 
                                               create_graph=True, retain_graph=True,)

            # dims = list(range(1, grad_real_full.ndim))
            # grad_crop_penalty_full = grad_real_full.pow(2).sum(dims) + grad_target_full.pow(2).sum(dims)
            # grad_crop_penalty_full *= (0.5 * self.opt.lambda_patch_R1 * 0.5)

            dims_feat_1 = list(range(1, grad_real_feat_1.ndim))
            grad_crop_penalty_feat_1 = grad_real_feat_1.pow(2).sum(dims_feat_1) + grad_target_feat_1.pow(2).sum(dims_feat_1)
            grad_crop_penalty_feat_1 *= (0.5 * self.opt.lambda_patch_R1 * 0.5)
        
            dims_feat_2 = list(range(1, grad_real_feat_2.ndim))
            grad_crop_penalty_feat_2 = grad_real_feat_2.pow(2).sum(dims_feat_2) + grad_target_feat_2.pow(2).sum(dims_feat_2)
            grad_crop_penalty_feat_2 *= (0.5 * self.opt.lambda_patch_R1 * 0.5)    
            
            dims_feat_3 = list(range(1, grad_real_feat_3.ndim))
            grad_crop_penalty_feat_3 = grad_real_feat_3.pow(2).sum(dims_feat_3) + grad_target_feat_3.pow(2).sum(dims_feat_3)
            grad_crop_penalty_feat_3 *= (0.5 * self.opt.lambda_patch_R1 * 0.5)

            dims_feat_4 = list(range(1, grad_real_feat_4.ndim))
            grad_crop_penalty_feat_4 = grad_real_feat_4.pow(2).sum(dims_feat_4) + grad_target_feat_4.pow(2).sum(dims_feat_4)
            grad_crop_penalty_feat_4 *= (0.5 * self.opt.lambda_patch_R1 * 0.5)

            dims_feat_5 = list(range(1, grad_real_feat_5.ndim))
            grad_crop_penalty_feat_5 = grad_real_feat_5.pow(2).sum(dims_feat_5) + grad_target_feat_5.pow(2).sum(dims_feat_5)
            grad_crop_penalty_feat_5 *= (0.5 * self.opt.lambda_patch_R1 * 0.5)

            grad_crop_penalty = (0.2 * grad_crop_penalty_feat_1) + (0.2 * grad_penalty_feat_2) + (0.2 * grad_penalty_feat_3) + (0.2 * grad_penalty_feat_4) + (0.2 * grad_penalty_feat_5) 
        else:
            grad_crop_penalty = 0.0

        losses["D_R1"] = grad_penalty + grad_crop_penalty

        return losses

    def compute_generator_losses(self, real, images_feat_1, images_feat_2, images_feat_3, images_feat_4, images_feat_5, sp_ma, gl_ma):
        losses, metrics = {}, {}
        B = real.size(0)
        mask_struct_1, mask_struct_2, mask_struct_3, mask_struct_4, mask_struct_5 = self.mask_maker()                
        sp, gl = self.E(real)
        sp_feat_1 = torch.mul(sp, mask_struct_1.to('cuda'))
        sp_feat_2 = torch.mul(sp, mask_struct_2.to('cuda'))
        sp_feat_3 = torch.mul(sp, mask_struct_3.to('cuda'))
        sp_feat_4 = torch.mul(sp, mask_struct_4.to('cuda'))
        sp_feat_5 = torch.mul(sp, mask_struct_5.to('cuda'))
        

        #print('sp.shape->', sp.shape, 'gl.shape->', gl.shape)
        
        #rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory
        rec_feat_1 = self.G(sp_feat_1[:B // 2], gl[:B // 2])
        rec_feat_2 = self.G(sp_feat_2[:B // 2], gl[:B // 2])
        rec_feat_3 = self.G(sp_feat_3[:B // 2], gl[:B // 2])
        rec_feat_4 = self.G(sp_feat_4[:B // 2], gl[:B // 2])
        rec_feat_5 = self.G(sp_feat_5[:B // 2], gl[:B // 2])
        
        sp_mix = self.swap(sp)
        sp_mix_feat_1 = self.swap(sp_feat_1)
        sp_mix_feat_2 = self.swap(sp_feat_2)
        sp_mix_feat_3 = self.swap(sp_feat_3)
        sp_mix_feat_4 = self.swap(sp_feat_4)
        sp_mix_feat_5 = self.swap(sp_feat_5)

        # record the error of the reconstructed images for monitoring purposes
        if self.opt.lambda_L1 > 0.0:
            #l1_dist_full = self.l1_loss(rec, real[:B // 2])
            l1_dist_feat_1 = self.l1_loss(rec_feat_1, images_feat_1[:B // 2])
            l1_dist_feat_2 = self.l1_loss(rec_feat_2, images_feat_2[:B // 2])
            l1_dist_feat_3 = self.l1_loss(rec_feat_3, images_feat_3[:B // 2])
            l1_dist_feat_4 = self.l1_loss(rec_feat_4, images_feat_4[:B // 2])
            l1_dist_feat_5 = self.l1_loss(rec_feat_5, images_feat_5[:B // 2])

            metrics["L1_dist"] = (0.2 * l1_dist_feat_1) + (0.2 * l1_dist_feat_2) + (0.2 * l1_dist_feat_3) + (0.2 * l1_dist_feat_4) + (0.2 * l1_dist_feat_5) 
            losses["G_L1"] = metrics["L1_dist"] * self.opt.lambda_L1
            train_metrics_to_log = {'gen_trg_l1_loss_feat_1': l1_dist_feat_1, 'gen_trg_l1_loss_feat_2': l1_dist_feat_2, 'gen_trg_l1_loss_feat_3': l1_dist_feat_3,
                                    'gen_trg_l1_loss_feat_4': l1_dist_feat_4, 'gen_trg_l1_loss_feat_5': l1_dist_feat_5}
            print(train_metrics_to_log)

        if self.opt.crop_size >= 1024:
            # another momery-saving trick: reduce #outputs to save memory
            # real = real[B // 2:]
            # gl = gl[B // 2:]
            # sp_mix = sp_mix[B // 2:]
            images_feat_1 = images_feat_1[B // 2:]
            gl_feat_1 = gl_feat_1[B // 2:]
            sp_mix_feat_1 = sp_mix_feat_1[B // 2:]

            images_feat_2 = images_feat_2[B // 2:]
            gl_feat_2 = gl_feat_2[B // 2:]
            sp_mix_feat_2 = sp_mix_feat_2[B // 2:]

            images_feat_3 = images_feat_3[B // 2:]
            gl_feat_3 = gl_feat_3[B // 2:]
            sp_mix_feat_3 = sp_mix_feat_3[B // 2:]

            images_feat_4 = images_feat_4[B // 2:]
            gl_feat_4 = gl_feat_4[B // 2:]
            sp_mix_feat_4 = sp_mix_feat_4[B // 2:]

            images_feat_5 = images_feat_5[B // 2:]
            gl_feat_5 = gl_feat_5[B // 2:]
            sp_mix_feat_5 = sp_mix_feat_5[B // 2:]

        mix = self.G(sp_mix, gl)
        mix_feat_1 = self.G(sp_mix_feat_1, gl)
        mix_feat_2 = self.G(sp_mix_feat_2, gl)
        mix_feat_3 = self.G(sp_mix_feat_3, gl)
        mix_feat_4 = self.G(sp_mix_feat_4, gl)
        mix_feat_5 = self.G(sp_mix_feat_5, gl)

        if self.opt.lambda_GAN > 0.0:
            
            #loss_g_gan_rec_full = loss.gan_loss(self.D(rec), should_be_classified_as_real=True)
            loss_g_gan_rec_feat_1 = loss.gan_loss(self.D(rec_feat_1), should_be_classified_as_real=True)
            loss_g_gan_rec_feat_2 = loss.gan_loss(self.D(rec_feat_2), should_be_classified_as_real=True)
            loss_g_gan_rec_feat_3 = loss.gan_loss(self.D(rec_feat_3), should_be_classified_as_real=True)
            loss_g_gan_rec_feat_4 = loss.gan_loss(self.D(rec_feat_4), should_be_classified_as_real=True)
            loss_g_gan_rec_feat_5 = loss.gan_loss(self.D(rec_feat_5), should_be_classified_as_real=True)
            
            losses["G_GAN_rec"] = ((0.2 * loss_g_gan_rec_feat_1) + (0.2 * loss_g_gan_rec_feat_2) + (0.2 * loss_g_gan_rec_feat_3) + (0.2 * loss_g_gan_rec_feat_4) + (0.2 * loss_g_gan_rec_feat_5))* (self.opt.lambda_GAN * 0.5) 
            # loss_g_gan_mix_full = loss.gan_loss(self.D(mix), should_be_classified_as_real=True)
            loss_g_gan_mix_feat_1 = loss.gan_loss(self.D(mix_feat_1), should_be_classified_as_real=True)
            loss_g_gan_mix_feat_2 = loss.gan_loss(self.D(mix_feat_2), should_be_classified_as_real=True)
            loss_g_gan_mix_feat_3 = loss.gan_loss(self.D(mix_feat_3), should_be_classified_as_real=True)
            loss_g_gan_mix_feat_4 = loss.gan_loss(self.D(mix_feat_4), should_be_classified_as_real=True)
            loss_g_gan_mix_feat_5 = loss.gan_loss(self.D(mix_feat_5), should_be_classified_as_real=True)

            losses["G_GAN_mix"] = ((0.2 * loss_g_gan_mix_feat_1) + (0.2 * loss_g_gan_mix_feat_2) + (0.2 * loss_g_gan_mix_feat_3) + (0.2 * loss_g_gan_mix_feat_4) + (0.2 * loss_g_gan_mix_feat_5)) * (self.opt.lambda_GAN * 1.0)

        if self.opt.lambda_PatchGAN > 0.0:
            # real_feat = self.Dpatch.extract_features(self.get_random_crops(real), aggregate=self.opt.patch_use_aggregation).detach()
            # mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))
            images_feat_1_feat = self.Dpatch.extract_features(self.get_random_crops(images_feat_1), aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat_1_feat = self.Dpatch.extract_features(self.get_random_crops(mix_feat_1))
            
            images_feat_2_feat = self.Dpatch.extract_features(self.get_random_crops(images_feat_2), aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat_2_feat = self.Dpatch.extract_features(self.get_random_crops(mix_feat_2))
            
            images_feat_3_feat = self.Dpatch.extract_features(self.get_random_crops(images_feat_3), aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat_3_feat = self.Dpatch.extract_features(self.get_random_crops(mix_feat_3))
            
            images_feat_4_feat = self.Dpatch.extract_features(self.get_random_crops(images_feat_4), aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat_4_feat = self.Dpatch.extract_features(self.get_random_crops(mix_feat_4))
            
            images_feat_5_feat = self.Dpatch.extract_features(self.get_random_crops(images_feat_5), aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat_5_feat = self.Dpatch.extract_features(self.get_random_crops(mix_feat_5))
            
            # loss_g_mix_full = loss.gan_loss(self.Dpatch.discriminate_features(real_feat, mix_feat), should_be_classified_as_real=True)
            loss_g_mix_feat_1 = loss.gan_loss(self.Dpatch.discriminate_features(images_feat_1_feat, mix_feat_1_feat), should_be_classified_as_real=True)
            loss_g_mix_feat_2 = loss.gan_loss(self.Dpatch.discriminate_features(images_feat_2_feat, mix_feat_2_feat), should_be_classified_as_real=True)
            loss_g_mix_feat_3 = loss.gan_loss(self.Dpatch.discriminate_features(images_feat_3_feat, mix_feat_3_feat), should_be_classified_as_real=True)
            loss_g_mix_feat_4 = loss.gan_loss(self.Dpatch.discriminate_features(images_feat_4_feat, mix_feat_4_feat), should_be_classified_as_real=True)
            loss_g_mix_feat_5 = loss.gan_loss(self.Dpatch.discriminate_features(images_feat_5_feat, mix_feat_5_feat), should_be_classified_as_real=True)

            losses["G_mix"] = ((0.2 * loss_g_mix_feat_1) + (0.2 * loss_g_mix_feat_2) + (0.2 * loss_g_mix_feat_3) + (0.2 * loss_g_mix_feat_4) + (0.2 * loss_g_mix_feat_5)) * self.opt.lambda_PatchGAN
        return losses, metrics

    def get_visuals_for_snapshot(self, img_feat_1):
        if self.opt.isTrain:
            # avoid the overhead of generating too many visuals during training
            img_feat_1 = img_feat_1[:2] if self.opt.num_gpus > 1 else img_feat_1[:4]
            #print(img_feat_1.shape)
        sp, gl = self.E(img_feat_1)
        layout = util.resize2d_tensor(util.visualize_spatial_code(sp), img_feat_1)
        rec = self.G(sp, gl)
        mix = self.G(sp, self.swap(gl))

        visuals = {"img_feat_1": img_feat_1, "layout": layout, "rec": rec, "mix": mix}
        return visuals

    def fix_noise(self, sample_image=None):
        """ The generator architecture is stochastic because of the noise
        input at each layer (StyleGAN2 architecture). It could lead to
        flickering of the outputs even when identical inputs are given.
        Prevent flickering by fixing the noise injection of the generator.
        """
        if sample_image is not None:
            # The generator should be run at least once,
            # so that the noise dimensions could be computed
            sp, gl = self.E(sample_image)
            self.G(sp, gl)
        noise_var = self.G.fix_and_gather_noise_parameters()
        return noise_var

    def encode(self, image, extract_features=False):
        return self.E(image, extract_features=extract_features)

    def decode(self, spatial_code, global_code):
        return self.G(spatial_code, global_code)

    def get_parameters_for_mode(self, mode):
        if mode == "generator":
            return list(self.G.parameters()) + list(self.E.parameters())
        elif mode == "discriminator":
            Dparams = []
            if self.opt.lambda_GAN > 0.0:
                Dparams += list(self.D.parameters())
            if self.opt.lambda_PatchGAN > 0.0:
                Dparams += list(self.Dpatch.parameters())
            return Dparams
