--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nn')
require('LayerNormalization')
require('nngraph')
require('base')
local ptb = require('data')
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('PTB benchmark')
cmd:text()
cmd:text('Options')
cmd:option('--model',   'small',   'small/medium/large')
cmd:option('--bn',      false,     'use recurrent BN')
cmd:option('--bn_all',  false,     'use recurrent BN on all three spots')
cmd:option('--ln',      false,     'use LN')
cmd:option('--ln_all',  false,     'use LN on all three spots')
cmd:option('--affine',  false,     'add affined transformation in BN/LN')
cmd:option('--eps',     1e-3,      'epsilon value in BN/LN')
cmd:text()

-- parse input params
cmd_params = cmd:parse(arg)

local params = nil

if cmd_params.model == 'large' then
    -- Train 1 day and gives 82 perplexity.
    params = {
        batch_size=20,
        seq_length=35,
        layers=2,
        decay=1.15,
        rnn_size=1500,
        dropout=0.65,
        init_weight=0.04,
        lr=1,
        vocab_size=10000,
        max_epoch=14,
        max_max_epoch=55,
        max_grad_norm=10
    }
elseif cmd_params.model == 'medium' then
    -- Train 0.5 day and give 
    params = {
        batch_size=20,
        seq_length=35,
        layers=2,
        decay=1 / 0.8,
        rnn_size=650,
        dropout=0.5,
        init_weight=0.05,
        lr=1,
        vocab_size=10000,
        max_epoch=6,
        max_max_epoch=39,
        max_grad_norm=5
    }
elseif cmd_params.model == 'small' then
    -- Trains 1h and gives test 115 perplexity.
    params = {
        batch_size=20,
        seq_length=20,
        layers=2,
        decay=2,
        rnn_size=200,
        dropout=0,
        init_weight=0.1,
        lr=1,
        vocab_size=10000,
        max_epoch=4,
        max_max_epoch=13,
        max_grad_norm=5
    }
elseif cmd_params.model == 'small_deep' then
    params = {
        batch_size=20,
        seq_length=20,
        layers=4,
        decay=2,
        rnn_size=200,
        dropout=0,
        init_weight=0.1,
        lr=1,
        vocab_size=10000,
        max_epoch=10,
        max_max_epoch=20,
        max_grad_norm=5
    }
else
    error(string.format('Unsupported model type: %s', cmd_params.model))
end
params.batch_norm     = cmd_params.bn
params.batch_norm_all = cmd_params.bn_all
params.layer_norm     = cmd_params.ln
params.layer_norm_all = cmd_params.ln_all
params.momentum       = 0.1
params.eps            = cmd_params.eps
params.affine         = cmd_params.affine

print(params)

local function transfer_data(x)
    return x:cuda()
end

local state_train, state_valid, state_test
local model = {}
local paramx, paramdx


function create_or_share(module_type, name, module_dict, args)
    local result
    if module_type == 'Linear' then
        if module_dict[name] then
            result = nn.Linear(unpack(args)):cuda()
            result:share(module_dict[name], 'weight', 'gradWeight', 'bias', 'gradBias')
            print('Share ' .. name)
        else
            result = nn.Linear(unpack(args)):cuda()
            module_dict[name] = result
            result.weight:uniform(-params.init_weight, params.init_weight)
            result.bias:zero()
            print('Create ' .. name)
        end
    elseif module_type == 'LookupTable' then
        if module_dict[name] then
            result = nn.LookupTable(unpack(args)):cuda()
            result:share(module_dict[name], 'weight', 'gradWeight')
            print('Share ' .. name)
        else
            result = nn.LookupTable(unpack(args)):cuda()
            module_dict[name] = result
            result.weight:uniform(-params.init_weight, params.init_weight)
            print('Create ' .. name)
        end
    elseif module_type == 'LayerNormalization' then
        if module_dict[name] then
            result = nn.LayerNormalization(unpack(args))
            result.module:cuda()
            if result.bias then
                result.bias:share(module_dict[name].bias, 'bias', 'gradBias')
            end
            if result.gain then
                result.gain:share(module_dict[name].gain, 'weight', 'gradWeight')
            end
            result = result.module
            print('Share ' .. name)
        else
            result = nn.LayerNormalization(unpack(args))
            result.module:cuda()
            module_dict[name] = result
            result = result.module
            print('Create ' .. name)
        end
    elseif module_type == 'BatchNormalization' then
        --if module_dict[name] then
            --result = nn.BatchNormalization(unpack(args)):cuda()
            --result:share(module_dict[name], 'weight', 'gradWeight', 'bias', 'gradBias')
            --print('Share ' .. name)
        --else
            result = nn.BatchNormalization(unpack(args)):cuda()
        --    module_dict[name] = result
            if result.bias then
                result.bias:zero()
            end
            if result.weight then
                result.weight = result.weight:zero() + 1
            end
            print('Create ' .. name)
        --end
    else
        error(string.format('Unsupported module type: %s', module_type))
    end
    return result
end

local module_dict = {}

local function lstm(x, prev_c, prev_h, layer_idx)
    -- Calculate all four gates in one go
    --local i2h              = nn.Linear(params.rnn_size, 4 * params.rnn_size)(x)
    local i2h              = create_or_share('Linear', 'i2h_' .. layer_idx,
                                               module_dict, 
                                               {params.rnn_size, 4 * params.rnn_size})(x)
    --local h2h              = nn.Linear(params.rnn_size, 4 * params.rnn_size)(prev_h)
    local h2h              = create_or_share('Linear', 'h2h_' .. layer_idx,
                                               module_dict,
                                               {params.rnn_size, 4 * params.rnn_size})(
                                               prev_h)
    if params.batch_norm and params.batch_norm_all then
        local eps          = params.eps
        local dim          = 4 * params.rnn_size
        local mom          = params.momentum
        local aff          = params.affine
        --local bni        = nn.BatchNormalization(dim, eps, mom, aff)
        --local bnh        = nn.BatchNormalization(dim, eps, mom, aff)
        local bni          = create_or_share('BatchNormalization', 'bni_' .. layer_idx,
                                             module_dict, {dim, eps, mom, aff})
        local bnh          = create_or_share('BatchNormalization', 'bnh_' .. layer_idx,
                                             module_dict, {dim, eps, mom, aff})
        bni:cuda()
        bnh:cuda()
        i2h                = bni(i2h)
        h2h                = bnh(h2h)
    elseif params.layer_norm and params.layer_norm_all then
        local eps          = params.eps
        local dim          = 4 * params.rnn_size
        local aff          = params.affine
        local lni          = create_or_share('LayerNormalization', 'lni_' .. layer_idx, 
                                             module_dict, {dim, 0, eps, aff})
        local lnh          = create_or_share('LayerNormalization', 'lnh_' .. layer_idx, 
                                             module_dict, {dim, 0, eps, aff})
        i2h                = lni(i2h)
        h2h                = lnh(h2h)
         
    end
    local gates            = nn.CAddTable()({i2h, h2h})

    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshaped_gates   = nn.Reshape(4,params.rnn_size)(gates)
    local sliced_gates     = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    if params.layer_norm then
        local eps          = params.eps
        local dim          = params.rnn_size
        local aff          = params.affine
        local ln           = create_or_share('LayerNormalization', 'lnc_'..layer_idx, 
                                             module_dict, {dim, 0, eps, aff})
        next_c = ln(next_c)
    elseif params.batch_norm then
        local eps          = params.eps
        local dim          = params.rnn_size
        local mom          = params.momentum
        local aff          = params.affine
        -- local bnc          = nn.BatchNormalization(dim, eps, mom, aff)
        local bnc          = create_or_share('BatchNormalization', 'bnc_' .. layer_idx,
                                             module_dict, {dim, eps, mom, aff})
        next_c = bnc(next_c)
    end
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    return next_c, next_h
end

local function create_network()
    local x                = nn.Identity()()
    local y                = nn.Identity()()
    local prev_s           = nn.Identity()()
    --local word_embed       = LookupTable(params.vocab_size, params.rnn_size)
    local word_embed       = create_or_share('LookupTable', 'word_embed',
                                             module_dict,
                                             {params.vocab_size, params.rnn_size})
    local i                = {[0] = word_embed(x)}
    word_embed.weight:uniform(-params.init_weight, params.init_weight)
    local next_s           = {}
    local split            = {prev_s:split(2 * params.layers)}
    for layer_idx = 1, params.layers do
        local prev_c         = split[2 * layer_idx - 1]
        local prev_h         = split[2 * layer_idx]
        local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
        local next_c, next_h = lstm(dropped, prev_c, prev_h, layer_idx)
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h
    end
    --local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
    local h2y              = create_or_share('Linear', 'h2y', module_dict,
                                             {params.rnn_size, params.vocab_size})
    h2y.weight:uniform(-params.init_weight, params.init_weight)
    h2y.bias:zero()
    local dropped          = nn.Dropout(params.dropout)(i[params.layers])
    local pred             = nn.LogSoftMax()(h2y(dropped))
    local err              = nn.ClassNLLCriterion()({pred, y})
    local module           = nn.gModule({x, y, prev_s},
    {err, nn.Identity()(next_s)})
    --module:getParameters():uniform(-params.init_weight, params.init_weight)
    return transfer_data(module)
    --return module
end

local function setup()
    print("Creating a RNN LSTM network.")
    local core_network, separate_modules = create_network()
    paramx, paramdx = core_network:getParameters()
    model.s = {}
    model.ds = {}
    model.start_s = {}
    for j = 0, params.seq_length do
        model.s[j] = {}
        for d = 1, 2 * params.layers do
            model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        end
    end
    for d = 1, 2 * params.layers do
        model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
    model.core_network = core_network
    
    model.rnns = {}
    for i = 1, params.seq_length do
        table.insert(model.rnns, create_network())
    end
    
    -- model.rnns = g_cloneManyTimes(core_network, params.seq_length)
    model.norm_dw = 0
    model.err = transfer_data(torch.zeros(params.seq_length))
    --model.rnns:cuda()
end

local function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

local function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

local function fp(state)
    g_replace_table(model.s[0], model.start_s)
    if state.pos + params.seq_length > state.data:size(1) then
        reset_state(state)
    end
    for i = 1, params.seq_length do
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
        state.pos = state.pos + 1
    end
    g_replace_table(model.start_s, model.s[params.seq_length])
    return model.err:mean()
end

local function bp(state)
    paramdx:zero()
    reset_ds()
    for i = params.seq_length, 1, -1 do
        state.pos = state.pos - 1
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        local derr = transfer_data(torch.ones(1))
        local tmp = model.rnns[i]:backward({x, y, s},
        {derr, model.ds})[3]
        g_replace_table(model.ds, tmp)
        cutorch.synchronize()
    end
    state.pos = state.pos + params.seq_length
    model.norm_dw = paramdx:norm()
    if model.norm_dw > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / model.norm_dw
        paramdx:mul(shrink_factor)
    end
    paramx:add(paramdx:mul(-params.lr))
end

local function run_valid()
    reset_state(state_valid)
    g_disable_dropout(model.rnns)
    local len = (state_valid.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    for i = 1, len do
        perp = perp + fp(state_valid)
    end
    ppl = g_f3(torch.exp(perp / len))
    print("Validation set perplexity : " .. ppl)
    g_enable_dropout(model.rnns)
    return ppl
end

local function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    ppl = g_f3(torch.exp(perp / (len - 1)))
    print("Test set perplexity : " .. ppl)
    g_enable_dropout(model.rnns)
    return ppl
end

local function main()
    g_init_gpu({})
    state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
    state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
    state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
    print("Network parameters:")
    print(params)
    local states = {state_train, state_valid, state_test}
    for _, state in pairs(states) do
        reset_state(state)
    end
    setup()
    local step = 0
    local epoch = 0
    local total_cases = 0
    local beginning_time = torch.tic()
    local start_time = torch.tic()
    print("Starting training.")
    local words_per_step = params.seq_length * params.batch_size
    local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
    local perps
    
    local logs_folder = "/u/mren/public_html/results/"
    local exp_id = nil
    local desc = nil
    if params.batch_norm then
        exp_id = "exp_PTB_bn_torch_"..os.date("%Y-%b-%d-%H-%M-%S")
        desc = "torch BN eps="..params.eps.." affine="..tostring(params.affine)
    elseif params.layer_norm then
        exp_id = "exp_PTB_ln_torch_"..os.date("%Y-%b-%d-%H-%M-%S")
        desc = "torch LN eps="..params.eps.." affine="..tostring(params.affine)
    else
        exp_id = "exp_PTB_baseline_torch_"..os.date("%Y-%b-%d-%H-%M-%S")
        desc = "torch baseline"
    end
    print("Experiment ID "..exp_id)
   
    -- Register logs
    local catalog_fn     = logs_folder.."catalog"
    local exp_folder     = logs_folder .. exp_id .. "/"
    local catalog_fn_exp = exp_folder .. "catalog"
    
    io.open(catalog_fn, "a"):write(exp_id..","..desc.."\n")
    os.execute("mkdir " .. exp_folder)

    local catalog_file = io.open(catalog_fn_exp, "w")
    catalog_file:write("filename,type,name\n")
    catalog_file:write("train_ppl.csv,csv,Train Perplexity\n")
    catalog_file:write("valid_ppl.csv,csv,Valid Perplexity\n")
    catalog_file:write("test_ppl.csv,csv,Test Perplexity\n")
    catalog_file:write("learn_rate.csv,csv,Learning Rate\n")
    catalog_file:close()

    train_ppl_fn  = exp_folder .. "train_ppl.csv"
    valid_ppl_fn  = exp_folder .. "valid_ppl.csv"
    test_ppl_fn   = exp_folder .. "test_ppl.csv"
    learn_rate_fn = exp_folder .. "learn_rate.csv"
    io.open(train_ppl_fn,  "w"):write("step,time,Train Perplexity\n")
    io.open(valid_ppl_fn,  "w"):write("step,time,Valid Perplexity\n")
    io.open(test_ppl_fn,   "w"):write("step,time,Test Perplexity\n")
    io.open(learn_rate_fn, "w"):write("step,time,Learning Rate\n")

    while epoch < params.max_max_epoch do
        local perp = fp(state_train)
        if perps == nil then
            perps = torch.zeros(epoch_size):add(perp)
        end
        perps[step % epoch_size + 1] = perp
        step = step + 1
        bp(state_train)
        total_cases = total_cases + params.seq_length * params.batch_size
        epoch = step / epoch_size
        if step % torch.round(epoch_size / 10) == 10 then
            local wps = torch.floor(total_cases / torch.toc(start_time))
            local since_beginning = g_d(torch.toc(beginning_time) / 60)
            train_perp = g_f3(torch.exp(perps:mean()))
            print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. train_perp ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')

            io.open(train_ppl_fn, "a"):write(step..","..os.time()..","..train_perp.."\n")
        end
        if step % epoch_size == 0 then
            valid_perp = run_valid()
            if epoch > params.max_epoch then
                params.lr = params.lr / params.decay
            end
            io.open(valid_ppl_fn, "a"):write(step..","..os.time()..","..valid_perp.."\n")
            io.open(learn_rate_fn, "a"):write(
                step..","..os.time()..","..params.lr.."\n")
            print("Experiment ID "..exp_id)
        end
        if step % 33 == 0 then
            cutorch.synchronize()
            collectgarbage()
        end
    end
    test_perp = run_test()
    io.open(test_ppl_fn, "a"):write(step..","..os.time()..","..test_perp.."\n")
    print("Training is over.")
end

main()
