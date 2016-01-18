require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'cutorch'
require 'cunn'

local LSTM = require 'LSTM'             -- LSTM timestep and utilities
local model_utils=require 'model_utils'

function round(x)
  return x>=0 and math.floor(x+0.5) or math.ceil(x-0.5)
end


torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple LSTM image plane labeling model.')
cmd:text()
cmd:text('Options')
cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-seq_length',1160,'number of timesteps to unroll to') --corresponds to the number of 20x20 windows in entire image (1160)
cmd:option('-rnn_size',32,'size of LSTM internal state')
cmd:option('-max_epochs',1,'number of full passes through the training data')
cmd:option('-savefile','model_autosave','filename to autosave the model (protos) to, appended with the,param,string.t7')
cmd:option('-save_every',100,'save every 100 steps, overwriting the existing file')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-noise',0,'multiplication factor for added noise')
cmd:option('-path','.','path to destination folder where results will be saved')

cmd:option('-input_size',400,'size of input vector, image patch reshaped as vector, default 20x20=400')
cmd:option('-start_val',1,'number of start validation set')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

trainSet=torch.load('train_set.t7')
trainSet.data=trainSet.data:cuda()
trainSet.labels=trainSet.labels:cuda()

path=opt.path

if(path:sub(path:len(),path:len()) ~= '/') then
	path=path ..'/'
end

-- preparation stuff:
torch.manualSeed(opt.seed)
--opt.savefile = cmd:string(opt.savefile, opt,{save_every=true, print_every=true, savefile=true, vocabfile=true, datafile=true}) .. '.t7'

classes={1,2,3,4,5,6,7,8,9,10,11,12}
local n_val=10

local n_shuffle
if(opt.start_val==1) then
	n_shuffle=torch.randperm(152) --new validation shuffle
	print('Saving shuffle file to '.. path..'noise_'.. round(opt.noise*100) ..'_shuffle.t7')
	torch.save(path..'noise_'.. round(opt.noise*100) ..'_shuffle.t7', n_shuffle)
else
	print('Loading shuffle file from '..path..'noise_'..round(opt.noise*100)..'_shuffle.t7')
	n_shuffle=torch.load(path..'noise_'.. round(opt.noise*100) ..'_shuffle.t7', n_shuffle)

end
--torch.save(path..'noise_'.. round(opt.noise*100) ..'_shuffle.t7', n_shuffle)

partitions={15,15,15,15,15,15,15,15,15,17}
partition_start={1,16,31,46,61,76,91,106,121,136}

for val_i=opt.start_val,n_val do 

	print('Model autosave path '..path..'model_autosave_val_' .. val_i ..'_noise_'.. round(opt.noise*100) ..'.t7')
	opt.savefile=path..'model_autosave_val_' .. val_i ..'_noise_'.. round(opt.noise*100) ..'.t7'

	local s_test=n_shuffle[{{partition_start[val_i],(partition_start[val_i]+partitions[val_i]-1)}}]
	local s_train=torch.Tensor(152-partitions[val_i]):zero()

	if(val_i==1) then
		s_train=n_shuffle[{{partition_start[2],152}}]
	elseif(val_i==10) then
		 s_train=n_shuffle[{{1,(partition_start[10]-1)}}]
	else
		s_train[{{1,(partition_start[val_i]-1)}}]=n_shuffle[{{1,(partition_start[val_i]-1)}}]
 		s_train[{{(partition_start[val_i]),(152-partitions[val_i])}}]=n_shuffle[{{(partition_start[val_i]+partitions[val_i]),152}}]

	end

	confusion = optim.ConfusionMatrix(classes)

	-- define model prototypes for ONE timestep, then clone them
	--
	protos = {} -- TODO: local
	protos.embed = nn.Sequential():add(nn.Linear(opt.input_size, opt.rnn_size)):add(nn.Tanh()):cuda()
	-- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
	protos.lstm = LSTM.lstm(opt):cuda()
	protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, 12 --[[nr. labels, 11 line labels, 1 bckg./noise]])):add(nn.LogSoftMax()):cuda()
	protos.criterion = nn.ClassNLLCriterion():cuda()

	-- put the above things into one flattened parameters tensor
	local params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.lstm, protos.softmax)
	params:uniform(-0.08, 0.08)

	-- make a bunch of clones, AFTER flattening, as that reallocates memory
	clones = {} -- TODO: local
	for name,proto in pairs(protos) do
	    print('cloning '..name)
	    clones[name] = model_utils.clone_many_times(proto, 2*opt.seq_length, not proto.parameters)
	end

	-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
	local initstate_c = torch.zeros(--[[opt.batch_size]]1, opt.rnn_size):cuda()
	local initstate_h = initstate_c:clone()

	-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
	local dfinalstate_c = initstate_c:clone()
	local dfinalstate_h = initstate_c:clone()

	-- do fwd/bwd and return loss, grad_params
	function feval(x)
	    if x ~= params then
		params:copy(x)
	    end
	    grad_params:zero()
	    
	    ------------------- forward pass -------------------
	    local embeddings = {}            -- input embeddings
	    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
	    local lstm_h = {[0]=initstate_h} -- output values of LSTM
	    local predictions = {}           -- softmax outputs
	    local loss = 0

	    local n_index=s_train[torch.randperm(152-20)[1]] -- choose next index; 1-132, leave 20 for test
		
            local inp_data=trainSet.data[{n_index,{}}]:clone()
	    local noise=torch.randn(1160,400):mul(opt.noise) --add approximately opt.noise of Gaussian noise
	    inp_data:add(noise:cuda())

	    -- 2*opt.seq_length, once iterate through images (20x20) at the input 
	    -- + once more just from embedding vector to produce labels
	    for t=1,2*opt.seq_length do 

		embeddings[t] = clones.embed[t]:forward( inp_data[t%1161+ (t>1160 and 1 or 0)]   ) 

		-- we're feeding the *correct* things in here, alternatively
		-- we could sample from the previous timestep and embed that, but that's
		-- more commonly done for LSTM encoder-decoder models
		lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})

		predictions[t] = clones.softmax[t]:forward(lstm_h[t])
		if t>opt.seq_length then
			loss = loss + clones.criterion[t]:forward(predictions[t], 1+trainSet.labels[{n_index,t-opt.seq_length}])
			confusion:add(predictions[t], 1+trainSet.labels[{n_index,t-1160}])
		end 
	    end
	    loss = loss / opt.seq_length

	    ------------------ backward pass -------------------
	    -- complete reverse order of the above
	    local dembeddings = {}                              -- d loss / d input embeddings
	    local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
	    local dlstm_h = {}                                  -- output values of LSTM

	    -- 2*opt.seq_length, once iterate back in time through labels at the output 
	    -- + once more just through the image data
	    for t=2*opt.seq_length,1,-1 do 
		
		if t>opt.seq_length then -- dodal
			local doutput_t = clones.criterion[t]:backward(predictions[t], 1+trainSet.labels[{n_index,t-opt.seq_length}])
			doutput_t:mul(trainSet.labels[{n_index,t-opt.seq_length}]>0 and 5 or 1)--0s are 5x more common
			dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)
		else	
			dlstm_h[t] = torch.FloatTensor(opt.rnn_size):zero() 
		end	

		-- backprop through LSTM timestep
		dembeddings[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
		    {embeddings[t], lstm_c[t-1], lstm_h[t-1]},
		    {dlstm_c[t], dlstm_h[t]}
		))

		-- backprop through embeddings
		clones.embed[t]:backward(inp_data[t%1161+ (t>1160 and 1 or 0)], dembeddings[t])
	    end

	    ------------------------ misc ----------------------
	    -- transfer final state to initial state (BPTT)
	    initstate_c:copy(lstm_c[#lstm_c])
	    initstate_h:copy(lstm_h[#lstm_h])

	    -- clip gradient element-wise
	    grad_params:clamp(-5, 5)

	    return loss, grad_params
	end

	-- optimization stuff
	losses = {} -- TODO: local
	local optim_state = {learningRate = 1e-1}
	local iterations = (152-20)*15--opt.max_epochs * loader.nbatches

	for i = 1, iterations do

	    if(i>0.5*iterations) --decrease learning rate
		then
			optim_state.learningRate=3*1e-2
		end

	    confusion:zero()
	    local _, loss = optim.adagrad(feval, params:cuda(), optim_state)
	    losses[#losses + 1] = loss[1]

	    if i % opt.save_every == 0 then
		torch.save(opt.savefile, protos)
	    end
	    if i % opt.print_every == 0 then
		print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], grad_params:norm()))
		print(confusion)
	    end
	end
end


