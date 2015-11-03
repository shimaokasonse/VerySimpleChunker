require "nn"
util = require("util")

-- MODEL
local bundle = torch.load("data/data.t7")
local model = torch.load("data/model.t7")
local word_encoder = bundle["word_encoder"]
local chunk_decoder = bundle["chunk_decoder"]
local padding_size = bundle["padding_size"]

function parse(file_name) 
    local f = io.open(file_name)
    local sentence = {}
    for line in f:lines() do
        local temp = {}
        for word in line:gmatch("%S+") do table.insert(temp, word) end
        local word = temp[1]

        if not word and sentence then
	    local sent_input = {}
            local words_with_ctx = util:pad(sentence,padding_size)
            for i = 1, #sentence do
                local inputs_word_with_ctx  = words_with_ctx[i]
		
                local data = {}
                for j = 1, #inputs_word_with_ctx do
                    table.insert(data,word_encoder:encode(inputs_word_with_ctx[j]))
                end
		table.insert(sent_input,data)
            end
	    _ , output = model:forward(torch.Tensor(sent_input)):max(2)
            for j = 1, output:size(1) do
	    	print(sentence[j].."\t"..chunk_decoder:decode(output[j][1]))
	    end
	    print("\n")
            sentence = {}
        end
        if word then
            table.insert(sentence,word)
        end
    end
    f:close()	
end

-- PARSE
parse(arg[1])
-- OUTPUT