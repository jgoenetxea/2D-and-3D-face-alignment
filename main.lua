require 'torch'
require 'nn'
require 'nngraph'
require 'paths'

require 'image'
require 'xlua'
local utils = require 'utils'
local opts = require 'opts'(arg)

-- Load optional libraries
xrequire('cunn')
xrequire('cudnn')

require 'cudnn'

-- Load optional data-loading libraries
xrequire('matio') -- matlab
npy4th = xrequire('npy4th') -- python numpy

torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

-- local fileList = utils.getFileList(opts)
local fileList = utils.getRectDefFileList(opts)
-- print(fileList)
local predictions = {}

local model = torch.load(opts.model)
local modelZ
if opts.type == '3D-full' then
    modelZ = torch.load(opts.modelZ)
    if opts.device ~= 'cpu' then modelZ = modelZ:cuda() end
    modelZ:evaluate()
end

if opts.device == 'gpu' then model = model:cuda() end
model:evaluate()



-- Test --
-- t = torch.Timer()
-- os.execute("sleep " .. 1)
-- print("First sleep spends "..t:time().real.." seconds.")
-- t = torch.Timer()
-- os.execute("sleep " .. 3)
-- print("First sleep spends "..t:time().real.." seconds.")
-- t = torch.Timer()
-- os.execute("sleep " .. 2)
-- print("First sleep spends "..t:time().real.." seconds.")
-- ---- --



for r = 1, #fileList do
	print('Proccess file '..fileList[r].fullname)
	local imageDirPath = fileList[r].directory.."/"..fileList[r].name
	print('Current image path: '..imageDirPath)
	-- Load rect definition file
	imageList = utils.loadRectDefFile(fileList[r].fullname)

	-- Store elapsed times
	eTimes = {}
	for i = 1, #imageList do
		-- os.exit()
		if imageList[i].use == "1" then
			-- For each file line load image and predict 
			local imageFilePath = imageDirPath.."/"..imageList[i].imageName
			local img = image.load(imageFilePath)
			if img:size(1)==1 then
				img = torch.repeatTensor(img, 3, 1, 1)
			end
			originalSize = img:size()
		
			img = utils.crop(img, imageList[i].center, imageList[i].scale, 256):view(1,3,256,256)
			if opts.device ~= 'cpu' then img = img:cuda() end

			local output = model:forward(img)[4]:clone()
			output:add(utils.flip(utils.shuffleLR(model:forward(utils.flip(img))[4])))
			timer2d = torch.Timer()
			local preds_hm, preds_img = utils.getPreds(output, imageList[i].center, imageList[i].scale)
			elapsed2Dpred = timer2d:time().real

			preds_hm = preds_hm:view(68,2):float()*4
			-- depth prediction
			if opts.type == '3D-full' then
				out = torch.zeros(68, 256, 256)
				for i=1,68 do
					if preds_hm[i][1] > 0 then
					    utils.drawGaussian(out[i], preds_hm[i], 2)
					end
				end
				out = out:view(1,68,256,256)
				local inputZ = torch.cat(img:float(), out, 2)
				if opts.device ~= 'cpu' then inputZ = inputZ:cuda() end
				timer3d = torch.Timer()
				local depth_pred = modelZ:forward(inputZ):float():view(68,1) 
				preds_hm = torch.cat(preds_hm, depth_pred, 2)
				elapsed3Dpred = timer3d:time().real
		--        preds_img = torch.cat(preds_img, depth_pred, 2)
			end

			elapsedTime = elapsed2Dpred + elapsed3Dpred
			print("Elapsed time: "..elapsedTime)
			table.insert(eTimes, elapsedTime)

			if opts.mode == 'demo' then
				-- fullname = opts.output..'/'..paths.basename(imageFilePath, '.'..paths.extname(imageFilePath))..'.txt'
				fullname = imageDirPath..'/'..paths.basename(imageFilePath, '.'..paths.extname(imageFilePath))..'.txt'
				print("Output file name: ", fullname)
				utils.export(fullname, preds_hm)
				-- utils.plot(img, preds_hm)
			end

			if opts.save then
				torch.save(opts.output..'/'..paths.basename(imageFilePath, '.'..paths.extname(imageFilePath))..'.t7', preds_img)
			end

		end -- if imageList[i].use == "1" then

--		if opts.mode == 'eval' then
--		    predictions[i] = preds_img:clone()+1.75
--		    xlua.progress(i,#fileList)
--		end
	end -- for i = 1, #imageList do

	-- Write the elapsed times to a file
	timeFileName = imageDirPath..'.times'
	print("Times written to: ", timeFileName)
	utils.saveElapsedTimes(timeFileName, eTimes)
end -- for r = 1, #fileList do

-- if opts.mode == 'eval' then
--     predictions = torch.cat(predictions,1)
--     local dists = utils.calcDistance(predictions,fileList)
--     utils.calculateMetrics(dists)
-- end
