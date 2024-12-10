
import torch
if __name__ == '__main__':
	import correlation
else:
	from models import correlation # the custom cost volume layer

def backwarp(tenInput, tenFlow, backwarp_tenGrid = {}, backwarp_tenPartial = {}):
	#plt.imshow(tenInput.detach().cpu().numpy()[0].transpose(1,2,0), cmap='gray')
	#plt.show()
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	if str(tenFlow.shape) not in backwarp_tenPartial:
		backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
	tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

	tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

	tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

	return tenOutput[:, :-1, :, :] * tenMask



class PWCNet(torch.nn.Module):
	def __init__(self, in_channels=3):
		super().__init__()
		self.in_channels = in_channels
		class Extractor(torch.nn.Module):
			def __init__(self, in_channels):
				super().__init__()

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tenInput):
				tenOne = self.moduleOne(tenInput)
				tenTwo = self.moduleTwo(tenOne)
				tenThr = self.moduleThr(tenTwo)
				tenFou = self.moduleFou(tenThr)
				tenFiv = self.moduleFiv(tenFou)
				tenSix = self.moduleSix(tenFiv)

				return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
			# end
		# end

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super().__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

				if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
				)
			# end

			def forward(self, tenOne, tenTwo, objPrevious):
				tenFlow = None
				tenFeat = None

				if objPrevious is None:
					tenFlow = None
					tenFeat = None
					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)
					tenFeat = torch.cat([ tenVolume ], 1)

				elif objPrevious is not None:
					tenFlow = self.moduleUpflow(objPrevious['tenFlow'])
					tenFeat = self.moduleUpfeat(objPrevious['tenFeat'])
					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)
					tenFeat = torch.cat([ tenVolume, tenOne, tenFlow, tenFeat ], 1)

				# end
				tenFeat = torch.cat([ self.moduleOne(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.moduleTwo(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.moduleThr(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.moduleFou(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.moduleFiv(tenFeat), tenFeat ], 1)
				tenFlow = self.moduleSix(tenFeat)

				return {
					'tenFlow': tenFlow,
					'tenFeat': tenFeat
				}
			# end
		# end

		class Refiner(torch.nn.Module):
			def __init__(self):
				super().__init__()

				self.moduleMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)
			# end

			def forward(self, tenInput):
				return self.moduleMain(tenInput)
			# end
		# end
		self.moduleExtractor = Extractor(self.in_channels)
	
		self.moduleTwo = Decoder(2)
		self.moduleThr = Decoder(3)
		self.moduleFou = Decoder(4)
		self.moduleFiv = Decoder(5)
		self.moduleSix = Decoder(6)

		self.moduleRefiner = Refiner()
		self.freeze()
	# end

	def forward(self, tenOne, tenTwo):
		tenOne = self.moduleExtractor(tenOne)
		tenTwo = self.moduleExtractor(tenTwo)

		objEstimate = self.moduleSix(tenOne[-1], tenTwo[-1], None)
		objEstimate = self.moduleFiv(tenOne[-2], tenTwo[-2], objEstimate)
		objEstimate = self.moduleFou(tenOne[-3], tenTwo[-3], objEstimate)	
		objEstimate = self.moduleThr(tenOne[-4], tenTwo[-4], objEstimate)	
		objEstimate = self.moduleTwo(tenOne[-5], tenTwo[-5], objEstimate)
		objEstimate = objEstimate['tenFlow'] + self.moduleRefiner(objEstimate['tenFeat'])	
		return objEstimate
	
	def freeze(self):
		for param in self.parameters():
			param.requires_grad = False

	def unfreeze(self):
		for param in self.parameters():
			param.requires_grad = True
	# end
# end

# if __name__ == '__main__':
# 	import numpy as np
# 	import matplotlib.pyplot as plt
# 	import cv2
# 	import flowiz as fz

# 	# img_1 = cv2.imread('/home/ywqqqqqq/Documents/github/NVISII/examples/output/001.png', cv2.IMREAD_UNCHANGED)
# 	# img_2 = cv2.imread('/home/ywqqqqqq/Documents/github/NVISII/examples/output/032.png', cv2.IMREAD_UNCHANGED)
# 	img_1 = np.load('/media/ywqqqqqq/文档/dataset/SpikeFlyingThings/dualflow_bugfree/000006/frame.npy')[5]
# 	img_2 = np.load('/media/ywqqqqqq/文档/dataset/SpikeFlyingThings/dualflow_bugfree/000006/frame.npy')[36]
# 	# flow = cv2.imread('/home/ywqqqqqq/sim_ws/src/rpg_esim/event_camera_simulator/esim_ros/img/3.265020_f.png', cv2.IMREAD_UNCHANGED)[:,:,:2]
# 	flow_f = (np.load('/media/ywqqqqqq/文档/dataset/SpikeFlyingThings/dualflow_bugfree/000006/flow_f.npy').astype(np.float64) - 2**15)/64.0
# 	t_flow_f = torch.FloatTensor(flow_f).permute(2,0,1)[None,...].cuda()
# 	flow_b = (np.load('/media/ywqqqqqq/文档/dataset/SpikeFlyingThings/dualflow_bugfree/000006/flow_b.npy').astype(np.float64) - 2**15)/64.0
# 	t_flow_b = torch.FloatTensor(flow_b).permute(2,0,1)[None,...].cuda()

# 	img_1 = torch.FloatTensor(img_1)[None,...].cuda()
# 	img_2 = torch.FloatTensor(img_2)[None,...].cuda()
# 	# flow = flow.astype(np.int16)
# 	# flow_x = (flow[:,:,0:1].astype(float)-2**15)/64.0
# 	# flow_y = (flow[:,:,1:2].astype(float)-2**15)/64.0

# 	spike = np.load('/media/ywqqqqqq/文档/dataset/SpikeFlyingThings/dualflow_bugfree/000006/spike.npy')
# 	spike_tmp = torch.zeros(1,32,256,448).cuda()
# 	for i in range(32):
# 		spike_tmp[:,i] = torch.Tensor(spike & 1)
# 		spike >>= 1

# 	spike_f = torch.zeros(1,32,256,448).cuda()
# 	spike_b = torch.zeros(1,32,256,448).cuda()
# 	for i in range(32):
# 		spike_f[:,i:i+1] = backwarp(spike_tmp[:,i:i+1], i*t_flow_f/31)
# 		spike_b[:,i:i+1] = backwarp(spike_tmp[:,i:i+1], (31-i)*t_flow_b/31)
# 	# flow = torch.FloatTensor(np.concatenate([flow_x, flow_y], 2)).permute(2,0,1)[None,...].cuda()


# 	# h1=np.hstack([img_1.repeat(1,3,1,1).cpu().numpy()[0].transpose(1,2,0), img_2.repeat(1,3,1,1).cpu().numpy()[0].transpose(1,2,0)])
# 	# h2=np.hstack([fz.convert_from_flow(flow_f), fz.convert_from_flow(flow_b)])
# 	# v = np.vstack([h1,h2])
# 	# v = v.astype(np.uint8)
# 	# plt.imshow(v)
# 	# plt.show()

		
# 	plt.subplot(2,2,1); 
# 	plt.imshow(img_1.cpu().numpy()[0].transpose(1,2,0)/255, 'gray'); 
# 	plt.subplot(2,2,2); 
# 	plt.imshow(img_2.cpu().numpy()[0].transpose(1,2,0)/255, 'gray'); 
# 	plt.subplot(2,2,3); 
# 	# plt.imshow(fz.convert_from_flow(flow_f))
# 	# plt.imshow(backwarp(img_2, t_flow_f)[0].cpu().numpy().transpose(1,2,0)/255); 
# 	plt.imshow(spike_f.mean(1).cpu().numpy()[0], 'gray'); 
# 	plt.subplot(2,2,4); 
# 	# plt.imshow(fz.convert_from_flow(flow_b))
# 	# plt.imshow(backwarp(img_1, t_flow_b)[0].cpu().numpy().transpose(1,2,0)/255)
# 	plt.imshow(spike_b.mean(1).cpu().numpy()[0], 'gray'); 
# 	plt.show()
# 	pass

if __name__ == '__main__':
	import cv2
	import numpy
	import math
	import flowiz as fz
	import matplotlib.pyplot as plt

	def img2ten(img):
		if len(img.shape) == 2:
			img = img[:,:,numpy.newaxis].repeat(3, axis=-1)
		intWidth = img.shape[1]
		intHeight = img.shape[0]
		intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
		intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

		ten = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(img.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))))
		ten = ten.cuda().view(1, 3, intHeight, intWidth)
		ten = torch.nn.functional.interpolate(input=ten, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
		return ten, intWidth, intHeight, intPreprocessedWidth, intPreprocessedHeight
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model = PWCNet().to(device)
	a = torch.load('./pretrained/pwcnet')
	model.load_state_dict(a)

	# img1 = cv2.imread('/media/ywqqqqqq/文档/dataset/REDS/REDS_120fps/train/train_orig/000/00000000.png')
	# img2 = cv2.imread('/media/ywqqqqqq/文档/dataset/REDS/REDS_120fps/train/train_orig/000/00000030.png')
	img1 = numpy.load('/media/ywqqqqqq/YWQ/Dataset/VidarCity/Simulated/SpikeFlyingThings/dualflow_large/000000/frame.npy')[0,0]
	img2 = numpy.load('/media/ywqqqqqq/YWQ/Dataset/VidarCity/Simulated/SpikeFlyingThings/dualflow_large/000000/frame.npy')[1,0]

	ten1, intWidth, intHeight, intPreprocessedWidth, intPreprocessedHeight = img2ten(img1)
	ten2, _, _, _, _ = img2ten(img2)
	# tenFlow = 20.0 * torch.nn.functional.interpolate(input=model(ten1, ten2), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	# tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	# tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	ten2 = torch.nn.functional.interpolate(input=ten2, size=(intHeight, intWidth), mode='bilinear', align_corners=False)
	# ten1_warp = backwarp(ten2, tenFlow)
	plt.subplot(2,3,1)
	plt.imshow(img1, 'gray')
	# plt.imshow(fz.convert_from_flow(tenFlow.detach().cpu().numpy()[0].transpose(1,2,0)))

	# tenFlow = 20.0 * torch.nn.functional.interpolate(input=model(ten2, ten1), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	# tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	# tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	# ten2 = torch.nn.functional.interpolate(input=ten2, size=(intHeight, intWidth), mode='bilinear', align_corners=False)
	plt.subplot(2,3,2)
	plt.imshow(img2, 'gray')
	# plt.imshow(fz.convert_from_flow(tenFlow.detach().cpu().numpy()[0].transpose(1,2,0)))

	plt.subplot(2,3,4)
	tenFlow = (numpy.load('/media/ywqqqqqq/YWQ/Dataset/VidarCity/Simulated/SpikeFlyingThings/dualflow_large/000000/flow_f.npy').astype(numpy.float32) - 2**15)/64.0
	plt.imshow(fz.convert_from_flow(tenFlow))
	# plt.imshow(backwarp(torch.FloatTensor(img2[None,None,...]).cuda(), torch.FloatTensor(tenFlow.transpose(2,0,1)[None,...]).cuda()).cpu().numpy()[0,0], 'gray')
	plt.subplot(2,3,5)
	tenFlow = (numpy.load('/media/ywqqqqqq/YWQ/Dataset/VidarCity/Simulated/SpikeFlyingThings/dualflow_large/000000/flow_b.npy').astype(numpy.float32) - 2**15)/64.0
	plt.imshow(fz.convert_from_flow(tenFlow))
	# plt.imshow(backwarp(torch.FloatTensor(img1[None,None,...]).cuda(), torch.FloatTensor(tenFlow.transpose(2,0,1)[None,...]).cuda()).cpu().numpy()[0,0], 'gray')
	
	plt.subplot(2,3,3)
	spike = numpy.load('/media/ywqqqqqq/YWQ/Dataset/VidarCity/Simulated/SpikeFlyingThings/dualflow_large/000000/spike.npy')
	s = numpy.zeros((43,720,1280))
	for i in range(43):
		s[i] = spike & 1
		spike >>= 1
	plt.imshow(s.mean(0),'gray')

	plt.subplot(2,3,6)
	plt.imshow(fz.convert_from_flow(tenFlow))
	plt.show()

	# cv2.imwrite('/media/ywqqqqqq/文档/dataset/REDS/REDS_120fps/test.png', fz.convert_from_flow(tenFlow.detach().cpu().numpy()[0].transpose(1,2,0)))

	print()


# end