local _M = {}

function _M:pad(sentence,padding_size)

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

return _M