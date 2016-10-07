function nn.LayerNormalization(nOutput, bias, eps, affine)
   local input = nn.Identity()()
   local mean = nn.Mean(2)(input)
   local mean_rep = nn.Replicate(nOutput,2)(mean) 

   local input_center = nn.CSubTable()({input, mean_rep})
   local std = nn.Sqrt()(nn.Mean(2)(nn.Square()(input_center)))
   local std_rep = nn.AddConstant(eps)(nn.Replicate(nOutput,2)(std))
   local output = nn.CDivTable()({input_center, std_rep})
   local biasTransform = nil
   local gainTransform = nil
   if affine then
       print('Affine!')
      biasTransform = nn.Add(nOutput, false)
      if bias ~=nil then
         biasTransform.bias:fill(bias)
      end
      gainTransform = nn.CMul(nOutput)
      gainTransform.weight:fill(1.)
      output = biasTransform(gainTransform(output))
   end
   local retval = {
       module = nn.gModule({input},{output}), 
       bias = biasTransform, 
       gain = gainTransform
   }
   return retval
end
