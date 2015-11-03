util = require("util")

function create_encoder_decoder(file_name,column_num, min_freq)
  
    ---- Processing File
    local f = io.open(file_name)
    local vocabulary = {}
    vocabulary["<PAD>"] = 10000
    vocabulary["</PAD>"] = 10000
    for line in f:lines() do
        local temp = {}
        for word in line:gmatch("%S+") do table.insert(temp, word) end
        local word = temp[column_num]
        if word then
            if vocabulary[word] then vocabulary[word] = vocabulary[word] + 1  else vocabulary[word] = 1 end
        end
    end
    f:close()
    
    ---- Creating Encoder and Decoder
    local id = 2
    local encoder = {}
    local decoder = {}
    for word,freq in pairs(vocabulary) do
        if freq > min_freq then 
            encoder[word] = id
            decoder[id] = word
            id = id + 1
        end
    end
    
    ---- Encoder's :encode method
    function encoder:encode(word) 
        local id = encoder[word]
        if id then return id else return 1 end
    end 
    
    ---- Decoder's :decode method
    function decoder:decode(id) 
        local word = decoder[id]
        if word then return word else return "$UNKNOWN$" end
    end
    
    return encoder, decoder
    
end
    

function create_dataset(file_name,padding_size,word_encoder,pos_encoder,chunk_encoder)
    local dataset = {}
    local f = io.open(file_name)
    local sentence = {}
    local sentence_pos = {}
    local sentence_iob = {}
    for line in f:lines() do
        local temp = {}
        for word in line:gmatch("%S+") do table.insert(temp, word) end
        local word = temp[1]
        local pos = temp[2]
        local iob = temp[3]
        
        if not word and sentence then 
            local words_with_ctx = util:pad(sentence,padding_size)
            for i = 1, #sentence do
                local inputs_word_with_ctx  = words_with_ctx[i]
                local data = {}
                for j = 1, #inputs_word_with_ctx do
                    table.insert(data,word_encoder:encode(inputs_word_with_ctx[j]))
                end
                local target_pos = pos_encoder:encode(sentence_pos[i])
                local target_iob = chunk_encoder:encode(sentence_iob[i])
                table.insert(data,target_pos)
                table.insert(data,target_iob)
                table.insert(dataset,data)
            end
            sentence = {} 
            sentence_pos = {}
            sentence_iob = {}
        end
        if word then
            table.insert(sentence,word)
            table.insert(sentence_pos,pos)
            table.insert(sentence_iob,iob)
        end
    end
    f:close()
    return dataset
end


--- MAIN
local train_test_filename = "data/train+test.txt" 
local train_filename = "data/train.txt"
local test_filename = "data/test.txt"
local padding_size = 3
local min_freq = 1
local word_encoder,word_decoder = create_encoder_decoder(train_test_filename ,1,min_freq)	
local pos_encoder,pos_decoder = create_encoder_decoder(train_test_filename ,2,min_freq)
local chunk_encoder,chunk_decoder = create_encoder_decoder(train_test_filename ,3,min_freq)
local train_dataset = torch.Tensor(create_dataset(train_filename,padding_size,word_encoder,pos_encoder,chunk_encoder))
local test_dataset = torch.Tensor(create_dataset(test_filename,padding_size,word_encoder,pos_encoder,chunk_encoder))
dataset = {}
dataset["train"] = train_dataset
dataset["test"] = test_dataset
dataset["word_encoder"] = word_encoder
dataset["word_decoder"] = word_decoder
dataset["pos_encoder"] = pos_encoder
dataset["pos_decoder"] = pos_decoder
dataset["chunk_encoder"] = chunk_encoder
dataset["chunk_decoder"] = chunk_decoder
dataset["padding_size"] = padding_size
torch.save("data/data.t7",dataset)
