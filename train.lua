require "nn"
require "optim"


-- Data
local data = torch.load("data/data.t7")
local train_dataset = data["train"]
local test_dataset = data["test"]
local voc_size = #(data["word_decoder"])
local chunk_size = #(data["pos_decoder"])
local context_size = data["padding_size"]*2 + 1
local dimension = 10

local train_inputs = train_dataset[{{},{1,context_size}}]
local train_target =  train_dataset[{{},{context_size+2}}]:select(2,1)

local test_inputs = test_dataset[{{},{1,context_size}}]
local test_target =  test_dataset[{{},{context_size+2}}]:select(2,1)

-- Model
local model = nn.Sequential()
model:add(nn.LookupTable(voc_size,dimension))
model:add(nn.Reshape(dimension * context_size))
model:add(nn.Linear(dimension * context_size,30))
model:add(nn.ReLU())
model:add(nn.Linear(30,chunk_size))
model:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()

local param, grad = model:getParameters()
param:uniform(-0.1, 0.1)

-- Train
local batch_size = 100
local epoch = 0
local config = {LearningRate = 0.01, Mormentum=0.9}
function train()
    epoch = (epoch or 0) + 1
    print("epoch"..epoch)
    for j = 1,train_inputs:size(1)-batch_size,batch_size do
        
        local batch_data = train_inputs[{{j,j+batch_size-1},{}}]
        local batch_labels = train_target[{{j,j+batch_size-1}}]
       
        local function feval(new_param)
            
            if param ~= new_param then
                param:copy(new_param)
            end
            
            grad:zero()
            
            local output = model:forward(batch_data)
    
            local loss = criterion:forward(output,batch_labels)
            local dldo = criterion:backward(output,batch_labels)
            model:backward(batch_data,dldo)
            return loss, grad
        end
        
        optim.sgd(feval,param,config)
    end
end



-- Test
function test()
    max_values,predictions = model:forward(test_inputs):max(2)
    err = 0
    predictions = predictions:byte()
    for i = 1,test_target:size(1) do
        if predictions[i][1]~= test_target[i] then
            err = err +1
        end
    end
    print("Precision: "..(test_target:size(1) - err )/test_target:size(1) )
 end
 
for i=1,500 do
    train()
    test()
end
torch.save("data/model.t7",model)