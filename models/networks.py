import torch
import torch.nn as nn
from torch.nn import init
import functools

####################
# initialize
####################

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    print('initializing [%s] ...' % classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print('initializing [%s] ...' % classname)
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        m.weight.data *= scale
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print('initializing [%s] ...' % classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

####################
# define network
####################
def create_model(opt):
    if opt['mode'] == 'sr' or opt['mode'] == 'sr_curriculum'or opt['mode'] == 'fi'or opt['mode'] == 'msan':
        netG = define_G(opt['networks']['G'])
        return netG
    elif opt['mode'] == 'srgan':
        netG = define_G(opt['networks']['G'])
        netD = define_D(opt['networks']['D'])
        return netG, netD
    else:
        raise NotImplementedError("The mode [%s] of networks is not recognized." % opt['mode'])

# Generator
def define_G(opt):
    which_model = opt['which_model'].lower()
    if which_model == 'sr_resnet_torch':
        import models.modules.archs as Arch
        netG = Arch.SRResNet(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], \
            num_blocks=opt['num_blocks'], upscale_factor=opt['scale'], norm_type=opt['norm_type'], mode=opt['mode'])
    elif which_model == 'sr_resnet':
        import models.modules.archs as Arch
        netG = Arch.SRResNet(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], \
            num_blocks=opt['num_blocks'], upscale_factor=opt['scale'], norm_type=opt['norm_type'], mode=opt['mode'],\
            upsample_mode='upconv')
    elif which_model == 'dbpn':
        import models.modules.archs as Arch
        # TODO: Need residual or only process on Y channel
        netG = Arch.DBPN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], \
                             bp_stages=opt['num_blocks'], upscale_factor=opt['scale'], norm_type=None,
                             mode=opt['mode'])
    elif which_model == 'd-dbpn':
        import models.modules.archs as Arch
        netG = Arch.D_DBPN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                         num_features=opt['num_features'], \
                         bp_stages=opt['num_blocks'], upscale_factor=opt['scale'], norm_type=None,
                         mode=opt['mode'])

    elif which_model == 'drbpn':
        import models.modules.drbpn_arch as DRBPNArch
        netG = DRBPNArch.DRBPN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_recurs=opt['num_recurs'])
    elif which_model == 'srcnn':
        import models.modules.srcnn_arch as SRCNNArch
        netG = SRCNNArch.SRCNN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])
    elif which_model == 'fsrcnn':
        import models.modules.fsrcnn_arch as FSRCNNArch
        netG = FSRCNNArch.FSRCNN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])
    elif which_model == 'vdsr':
        import models.modules.vdsr_arch as VDSRArch
        netG = VDSRArch.VDSR(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])
    elif which_model == 'drcn':
        import models.modules.drcn_arch as DRCNArch
        netG = DRCNArch.DRCN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])
    elif which_model == 'drrn':
        import models.modules.drrn_arch as DRRNArch
        netG = DRRNArch.DRRN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])
    elif which_model == 'lapsrn':
        import models.modules.lapsrn_arch as LAPSRNArch
        netG = LAPSRNArch.LAPSRN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])
    elif which_model == 'carn':
        import models.modules.carn_arch as CARNNArch
        netG = CARNNArch.CARN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])
    elif which_model == 'msrn':
        import models.modules.msrn_arch as MSRNArch
        netG = MSRNArch.MSRN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])
    elif which_model == 'edsr':
        import models.modules.edsr_arch as EDSRArch
        netG = EDSRArch.EDSR(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                              num_features=opt['num_features'], num_recurs=opt['num_recurs'],
                              upscale_factor=opt['scale'])
    elif which_model == 'oisr_rks':
        import models.modules.oisr_rks_arch as OISR_RKSArch
        netG = OISR_RKSArch.OISR_RKS(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                              num_features=opt['num_features'], num_recurs=opt['num_recurs'],
                              upscale_factor=opt['scale'])

    elif which_model == 'rcan':
        import models.modules.rcan_arch as RCANArch
        netG = RCANArch.RCAN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                              num_features=opt['num_features'], num_recurs=opt['num_recurs'],
                              upscale_factor=opt['scale'])
    elif which_model == 'srmdnf':
        import models.modules.srmdnf_arch as SRMDNFArch
        netG = SRMDNFArch.SRMDNF(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                              num_features=opt['num_features'], num_recurs=opt['num_recurs'],
                              upscale_factor=opt['scale'])
    elif which_model == 'tsan':
        import models.modules.tsan_arch as TSANArch
        netG = TSANArch.TSAN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                                 num_features=opt['num_features'], num_recurs=opt['num_recurs'],
                                 upscale_factor=opt['scale'])
    elif which_model == 'realsr':
        import models.modules.realsr_arch as RealSRArch
        netG = RealSRArch.RealSR(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                                 num_features=opt['num_features'], num_recurs=opt['num_recurs'],
                                 upscale_factor=opt['scale'])


    elif which_model == 'aan':
        import models.modules.aan_arch as AANArch
        netG = AANArch.AAN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_recurs=opt['num_recurs'],
                             upscale_factor=opt['scale'])

    elif which_model == 'aaf':
        import models.modules.aaf_arch as AAFArch
        netG = AAFArch.AAF(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                           num_features=opt['num_features'], num_recurs=opt['num_recurs'],
                           upscale_factor=opt['scale'])
    elif which_model == 'mswsr':
        import models.modules.mswsr_arch as MSWSRArch
        netG = MSWSRArch.MSWSR(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                           num_features=opt['num_features'], num_recurs=opt['num_recurs'],
                           upscale_factor=opt['scale'])
    elif which_model == 'acnet':
        import models.modules.acnet_arch as ACNETArch
        netG = ACNETArch.ACNet(upscale_factor=opt['scale'])
    elif which_model == 'usrnet':
        import models.modules.usrnet_arch as USRNETArch
        netG = USRNETArch.USRNet(upscale_factor=opt['scale'],in_channels=opt['in_channels'], out_channels=opt['out_channels'])

    elif which_model == 'pan':
        import models.modules.pan_arch as PANArch
        netG =PANArch.PAN(upscale_factor=opt['scale'])




    elif which_model == 'fi':
        import models.modules.PAM_CAN__weight1 as FIArch
        netG = FIArch.FI(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                               num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])
    elif which_model == 'msan':
        import models.modules.msan_arch62 as MSANArch
        netG = MSANArch.MSAN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                               num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])
    elif which_model == 'ipt':
        import models.modules.ipt_arch3 as IPTArch
        netG = IPTArch.IPT(patch_size=opt['patch_size'],in_channels=opt['in_channels'], out_channels=opt['out_channels'], task_id = opt['task_id'],
                            depth = opt['depth'])
    elif which_model == 'usr':
        import models.modules.USR_arch1 as USRArch
        netG =USRArch.MSAN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                               num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])
    elif which_model == 'dct':
        import models.modules.dct_arch64 as DCTArch
        netG =DCTArch.DCT(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                               num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])




    elif which_model == 'iptw':
        import models.modules.iptw_arch as IPTWArch
        netG = IPTWArch.ImageProcessingTransformer(patch_size=opt['patch_size'], in_channels=opt['in_channels'],
                           out_channels=opt['out_channels'], task_id=opt['task_id'],
                           depth=opt['depth'])




    elif which_model == 'drudn':
        import models.modules.drudn_arch as DRUDNArch
        netG = DRUDNArch.DRUDN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                               num_features=opt['num_features'], num_recurs=opt['num_recurs'], upscale_factor=opt['scale'])


    elif which_model.find('feedback') >= 0:
        import models.modules.feedback_arch as FeedbackArch
        netG = FeedbackArch.FeedbackNetBaseline(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                                        num_features=opt['num_features'], num_steps=opt['num_steps'], num_units=opt['num_units'], num_stages=opt['num_stages'],
                                        upscale_factor=opt['scale'])
        # netG = FeedbackArch.FeedbackNetBaseline(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
        #                                 num_features=opt['num_features'], num_steps=opt['num_steps'],
        #                                 upscale_factor=opt['scale'])
    elif which_model == 'feedback_lstm':
        import models.modules.feedback_modules.feedbacknet as FeedbackLSTMArch
        netG = FeedbackLSTMArch.FeedbackNet(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                                        num_features=opt['num_features'], num_steps=opt['num_steps'], num_iterations=opt['num_iterations'],
                                        upscale_factor=opt['scale'])
    elif which_model == 'conv_test':
        import models.modules.archs as Arch
        netG = Arch.ConvTest(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], \
            num_blocks=opt['num_blocks'], upscale_factor=opt['scale'], norm_type=opt['norm_type'], mode=opt['mode'])

    elif which_model == 'sr_explore':
        import models.modules.sr_explore_arch as sr_explore_arch
        netG = sr_explore_arch.SRCNN3group_linear(opt['nf'])
    elif which_model.find('rcan') >= 0:
        import models.modules.rcan_arch as RCANArch
        netG = RCANArch.RCAN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_groups = opt['num_groups'], num_blocks = opt['num_blocks'], upscale_factor=opt['scale'])
    else:
        raise NotImplementedError("Network [%s] is not recognized." % which_model)

    if torch.cuda.is_available():
        netG = nn.DataParallel(netG).cuda()

    return netG

# Discriminator
def define_D(opt):
    # TODO: modify key value
    which_model = opt['which_model']

    if which_model == 'discriminaotr_vgg_128':
        import models.modules.archs as Arch
        netD = Arch.Discriminaotr_VGG_128(in_channels=opt['in_channels'], base_nf=opt['nf'], \
            norm_type=opt['norm_type'], mode=opt['mode'] ,act_type=opt['act_type'])
    elif which_model == 'discriminaotr_vgg_32':
        import models.modules.archs as Arch
        netD = Arch.Discriminaotr_VGG_32(in_channels=opt['in_channels'], base_nf=opt['nf'], \
            norm_type=opt['norm_type'], mode=opt['mode'] ,act_type=opt['act_type'])
    elif which_model == 'discriminaotr_vgg_32_y':
        import models.modules.archs as Arch
        netD = Arch.Discriminaotr_VGG_32_Y(in_channels=opt['in_channels'], base_nf=opt['nf'], \
            norm_type=opt['norm_type'], mode=opt['mode'] ,act_type=opt['act_type'])

    elif which_model == 'fsd':
        import models.modules.archs as Arch
        netD = Arch.DiscriminatorBasic(in_channels=3)
        print('# Initializing FSSR-DiscriminatorBasic with {} norm layer'.format(opt['norm_type']))

    elif which_model == 'nld_s2':
        import models.modules.archs as Arch
        netD = Arch.NLayerDiscriminator(in_channels=opt['in_channels'], ndf=64, n_layers=2, norm_type=opt['norm_type'], stride=2)
        print('#Initializing NLayer-Discriminator-stride-2 with {} norm-layer'.format(opt['norm_type'].upper()))

    elif which_model == 'nld_s1':
        import models.modules.archs as Arch
        netD = Arch.NLayerDiscriminator(in_channels=opt['in_channels'], ndf=64, n_layers=2, norm_type=opt['norm_type'], stride=1)
        print('#Initializing NLayer-Discriminator-stride-2 with {} norm-layer'.format(opt['norm_type'].upper()))
    else:
        raise NotImplementedError('Discriminator model [%s] is not recognized' % which_model)

    # init_weights(netD, init_type='kaiming', scale=1)
    if torch.cuda.is_available():
        netD = nn.DataParallel(netD).cuda()
    return netD


def define_F(opt, use_bn=False):
    import models.modules.archs as Arch
    gpu_ids = opt['gpu_ids']
    tensor = torch.cuda.FloatTensor if gpu_ids else torch.FloatTensor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = Arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, tensor=tensor)
    if gpu_ids:
        netF = nn.DataParallel(netF).cuda()
    netF.eval()  # No need to train
    return netF
