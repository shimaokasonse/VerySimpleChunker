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
    
function pad(sentence,padding_size)
    
    local sent_with_padding = {}
    local j = 1
    for i = 1,padding_size do
        sent_with_padding[j] = "<PAD>"
        j = j + 1
    end
    for i = 1, #sentence do
        sent_with_padding[j] = sentence[i]
        j = j + 1
    end
     for i = 1,padding_size do
        sent_with_padding[j] = "</PAD>"
        j = j + 1
    end
    
    result = {}
    
    for i = padding_size + 1, #sentence + padding_size do
        local temp = {}
        for k = -padding_size,padding_size do
            table.insert(temp,sent_with_padding[i+k])
        end
        table.insert(result, temp)
    end
    return result
end

function create_dataset(file_name,padding_size)
    local word_encoder,word_decoder = create_encoder_decoder(file_name,1,0)
    local pos_encoder,pos_decoder = create_encoder_decoder(file_name,2,0)
    local chunk_encoder,chunk_decoder = create_encoder_decoder(file_name,3,0)

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
            local words_with_ctx = pad(sentence,padding_size)
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

dataset = torch.Tensor(create_dataset("train.txt",2))
torch.save("data.t7",dataset)
