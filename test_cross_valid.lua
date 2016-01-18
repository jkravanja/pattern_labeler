require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
local LSTM = require 'LSTM'

function round(x)
  return x>=0 and math.floor(x+0.5) or math.ceil(x-0.5)
end

cmd=torch.CmdLine()
cmd:text()
cmd:text('Test')
cmd:text()

cmd:option('-model_dir','', 'Path to directory with model and shuffle file')
cmd:option('-output_dir','', 'Path to output directory')
cmd:option('-train_set','train_set_unnormalized.t7', 'Path to training set')
cmd:option('-noise',25,'multiplication factor for added noise')
cmd:text()
opt=cmd:parse(arg)

path=opt.model_dir
if(path:sub(path:len(),path:len()) ~= '/') then
	path=path ..'/'
end

shuffleFile='noise_'.. round(opt.noise*100) ..'_shuffle.t7'

optput_file=opt.output_dir--'results_test_labels/'
if(optput_file:sub(optput_file:len(),optput_file:len()) ~= '/') then
	optput_file=optput_file ..'/'
end

partitions={15,15,15,15,15,15,15,15,15,17}
partition_start={1,16,31,46,61,76,91,106,121,136}

torch.setdefaulttensortype('torch.FloatTensor')

print('Loading train/data set '..opt.train_set)
trainSet=torch.load(opt.train_set) --shuffled_train_set.t7

--get mean,std
mean=torch.zeros(400);
std=torch.zeros(400);
for i=1,400 do mean[i]=trainSet.data[{{},{}, i}]:mean() end
for i=1,400 do std[i]=trainSet.data[{{},{}, i}]:std() end

print('Loading shufflie file '..path..shuffleFile)
n_shuffle=torch.load(path..shuffleFile) 
opt.rnn_size = 32 --protos.embed.weight:size(2)
opt.length=1160

-- LSTM initial state, note that we're using minibatches OF SIZE ONE here
local prev_c = torch.zeros(1, opt.rnn_size)
local prev_h = prev_c:clone()
classes={1,2,3,4,5,6,7,8,9,10,11,12}
confusion = optim.ConfusionMatrix(classes)
confusion_all = optim.ConfusionMatrix(classes)
confusion_all:zero()
for val_i=1,10
do

	proto_file=path..'model_autosave_val_' .. val_i ..'_noise_'.. round(opt.noise*100) ..'.t7'
	print('Cross-validation set '..val_i)
	print('Loading model file '..proto_file)
	protos = torch.load(proto_file) 

	local s_test=n_shuffle[{{partition_start[val_i],(partition_start[val_i]+partitions[val_i]-1)}}]
	
	for inde=1,s_test:size(1) do
	
		confusion:zero()

		n_index=s_test[inde]
		F=torch.DiskFile(optput_file..n_index..'.txt','w')

		--now start sampling/argmaxing
		for t=1, 2*opt.length do
		    -- embedding and LSTM 
		    local inp_data=trainSet.data[{n_index,t%1161+ (t>1160 and 1 or 0)}]:clone()

		    local noise=torch.randn(400):mul(opt.noise) --add opt.noise of Gaussian noise

		    inp_data:add(noise)

		    --normalize
			for i=1,400 do
				inp_data[i]=inp_data[i]-mean[i]
				inp_data[i]=inp_data[i]/std[i]
			end

		    local embedding = protos.embed:forward(inp_data)
		    local next_c, next_h = unpack(protos.lstm:forward{embedding, prev_c, prev_h})
		    prev_c:copy(next_c)
		    prev_h:copy(next_h)
		    
		    if t>opt.length
		    then
			    -- softmax from previous timestep
			    local log_probs = protos.softmax:forward(next_h)

			    confusion:add(log_probs, 1+trainSet.labels[{n_index,t-1160}])
			    confusion_all:add(log_probs, 1+trainSet.labels[{n_index,t-1160}])

			    local _, amax = log_probs:max(1)
			    F:writeFloat(amax[1])
		    end
		end
		collectgarbage()
		F:close()
		print(confusion)
	end
end
print("==========Common confusion matrix==========")
print(confusion_all)


