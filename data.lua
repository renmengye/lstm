--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local file = require('pl.file')

local ptb_path = "./data/"

local vocab_idx = 0
local vocab_map = {}

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
local function replicate(x_inp, batch_size)
   print(x_inp:size())
   local s = x_inp:size(1)
   local x = torch.zeros(torch.floor(s / batch_size), batch_size)
   for i = 1, batch_size do
     local start = torch.round((i - 1) * s / batch_size) + 1
     local finish = start + x:size(1) - 1
     print('Start')
     print(start)
     print('Finish')
     print(finish)
     x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
   end
   return x
end

local function load_data(fname, build_vocab)
   local data = file.read(fname)
   data = stringx.replace(data, '\n', '<eos>')
   data = stringx.split(data)
   print(string.format("Loading %s, size of data = %d", fname, #data))
   local x = torch.zeros(#data)
   for i = 1, #data do
      local idx = -1
      if vocab_map[data[i]] == nil then
        if build_vocab then
         vocab_idx = vocab_idx + 1
         vocab_map[data[i]] = vocab_idx
         idx = vocab_map[data[i]]
        end
      else
        idx = vocab_map[data[i]]
      end
      if idx > -1 then
        x[i] = idx
      end
   end
   if build_vocab then
     local vocab_file = io.open("vocab.csv", "w")
     for key, value in pairs(vocab_map) do
      vocab_file:write(key..","..value.."\n")
     end
     vocab_file:close()
   end
   return x
end

local function traindataset(batch_size)
   local x = load_data(ptb_path .. "ptb.train.txt", true)
   x = replicate(x, batch_size)
   return x
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset(batch_size)
   local x = load_data(ptb_path .. "ptb.test.txt", false)
   x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
   return x
end

local function validdataset(batch_size)
   local x = load_data(ptb_path .. "ptb.valid.txt", false)
   x = replicate(x, batch_size)
   return x
end

return {traindataset=traindataset,
        testdataset=testdataset,
        validdataset=validdataset}
