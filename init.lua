require 'torch'
require 'utils'

local ffi = require 'ffi'

ffi.cdef[[
typedef struct 
{

    unsigned int *** doc;
    unsigned int ** line;
    unsigned int * full_idx;

    unsigned int * doc_id;
    unsigned int * doc_size;
    unsigned int ** line_size;
    unsigned int * full_line_size;

    unsigned int ndoc;
    unsigned int nline;
    unsigned int ntoken;
} Wikidoc;

void *malloc(size_t size);
void free(void *ptr);

]]

-- define ctype
c_short = ffi.typeof("unsigned short")
c_float = ffi.typeof("float")
c_int = ffi.typeof("unsigned int")
c_p_int = ffi.typeof("unsigned int*")
c_p_float = ffi.typeof("float*")
c_pp_int = ffi.typeof("unsigned int**")
c_ppp_int = ffi.typeof("unsigned int***")
c_p_inttensor = ffi.typeof("THIntTensor*")
c_p_intstorage = ffi.typeof("THIntStorage*")
c_p_floatstorage = ffi.typeof("THFloatStorage*")
c_p_longtensor = ffi.typeof("THLongTensor*")
c_p_longstorage = ffi.typeof("THLongStorage*")
struct = ffi.typeof("Wikidoc")
p_struct = ffi.typeof("Wikidoc*")


-- scan files into directory
function scandir(path)
   local i, t = 0, {}
   for filename in io.popen("ls "..path):lines() do
      i = i + 1
      t[i] = filename
   end
   return t
end


wikidocu = {}
setmetatable(wikidocu, {
        __call = function(self, filename, threshold, weight, verbose)

                     local data = {}
                     local weight = weight or false
                     local verbose = verbose or false
                     -- get size
                     local ndoc, nline, totallength = wikidocu.size(filename, threshold, verbose)

                     -- create C array
                     -- for indices
                     local ptrdoc = ffi.cast(c_ppp_int, ffi.C.malloc(ffi.sizeof(c_pp_int)*ndoc))
                     local ptrline = ffi.cast(c_pp_int, ffi.C.malloc(ffi.sizeof(c_p_int)*nline))
                     local ptrfull_idx = ffi.cast(c_p_int, ffi.C.malloc(ffi.sizeof(c_int)*totallength))
                     -- for size
                     local ptrdoc_size  = ffi.cast(c_p_int, ffi.C.malloc(ffi.sizeof(c_int)*ndoc))
                     local ptrline_size  = ffi.cast(c_pp_int, ffi.C.malloc(ffi.sizeof(c_p_int)*ndoc))
                     local ptrfull_line_size  = ffi.cast(c_p_int, ffi.C.malloc(ffi.sizeof(c_int)*nline))
                     -- for id
                     local ptrdoc_id  = ffi.cast(c_p_int, ffi.C.malloc(ffi.sizeof(c_int)*ndoc))

                     -- load data
                     wikidocu.load(ptrfull_idx, ptrfull_line_size, ptrline, ptrdoc, ptrdoc_id, ptrdoc_size, ptrline_size, filename, threshold, verbose)

                     -- create Wikidoc struct
                     local list = ffi.cast(p_struct, ffi.C.malloc(ffi.sizeof(struct)))
                     list.ndoc = ndoc
                     list.nline = nline
                     list.ntoken = totallength
                     list.full_idx = ptrfull_idx
                     list.full_line_size = ptrfull_line_size
                     list.line = ptrline
                     list.doc = ptrdoc
                     list.doc_id = ptrdoc_id
                     list.doc_size = ptrdoc_size
                     list.line_size = ptrline_size

                     function data:free()
                        ffi.C.free(ptrfull_idx)
                        ffi.C.free(ptrline)
                        ffi.C.free(ptrdoc)
                        ffi.C.free(ptrfull_line_size)
                        ffi.C.free(ptrdoc_id)
                        ffi.C.free(ptrdoc_size)
                        ffi.C.free(ptrline_size)
                        ffi.C.free(list)
                     end

                     function data:weight()
                        local s = torch.IntStorage()
                        local ptr = ffi.cast(ffi.typeof('THIntStorage*'), torch.pointer(s))
                        ptr.flag = 1
                        ptr.data = list.doc_size
                        ptr.size = list.ndoc

                        local prob = torch.IntTensor(s)
                        local ratio = 1./prob:sum()

                        return prob:double()*ratio
                     end

                     -- create sampler if needed
                     if weight then
                        walker = require 'walker'
                        local w = data:weight()
                        sampler = walker(w)
                     end

      
                     function data:doc(index)
                        local Cindex = index-1
                        local d = list.doc[Cindex]
                        local dlen = list.doc_size[Cindex]
                        local sizes = torch.IntStorage()
                        local ptr_sz = ffi.cast(c_p_intstorage, torch.pointer(sizes))
                        ptr_sz.size = dlen
                        ptr_sz.flag = 1
                        ptr_sz.data = list.line_size[Cindex]

                        -- cumulative sum
                        local positions = torch.IntTensor(dlen):zero()
                        if dlen>1 then
                           positions:narrow(1,2,dlen-1):copy(torch.IntTensor(sizes):cumsum(1):narrow(1,1,dlen-1))
                        end   
                        for i=1,positions:size(1) do positions[i]=positions[i]+1 end
  --                      positions:add(1)
                        
                        local totallen = positions[positions:nElement()]-1+sizes[dlen]
                        local storage = torch.IntStorage()
                        ptr_s = ffi.cast(c_p_intstorage, torch.pointer(storage))
                        ptr_s.size = totallen
                        ptr_s.flag = 1
                        ptr_s.data = list.doc[Cindex][0]

                        local idoc = {id=list.doc_id[Cindex]}

                        function idoc:size()
                           return dlen
                        end
                        function idoc:total()
                           return totallen
                        end

                        setmetatable(idoc, {__index = function(self, idx)
                                                         local position = positions[idx]
                                                         return torch.IntTensor(storage, position, sizes[idx])
                                                      end})
                        return idoc
                     end

                     function data:minibatch(size,id,x,y)
                        local docid = id or torch.Tensor(size)
                        local ngram = x or torch.Tensor(size*threshold)
                        local nextword = y or torch.Tensor(size)

                        if ngram:type()~="torch.IntTensor" then
                           ngram = ngram:int()
                        end

                        local randperm
                        if weight then
                           randperm = torch.Tensor(size)
                           for i=1,size do
                              randperm[i] = sampler()
                           end
                        else
                           randperm = torch.randperm(list.ndoc):narrow(1,1,size)
                        end

                        local ptr_ngram = ngram:data()
                        for i=1,size do
                           local k = randperm[i]-1
                           -- store docid
                           docid[i] = list.doc_id[k]
                           -- get random line from that doc
                           local l = math.random(list.doc_size[k])-1
                           -- get random ngram from that line
                           local n = math.random(list.line_size[k][l]-threshold)-1
                           ffi.copy(ptr_ngram+(i-1)*threshold,list.doc[k][l]+n,threshold*ffi.sizeof(c_int))
                           nextword[i] = list.doc[k][l][n+threshold]
                        end
                        return docid,ngram,nextword
                     end

                     function data:sample(t,size)
                        local ngram = t:resize(size)
                        if ngram:type()~="torch.IntTensor" then
                           ngram = ngram:int()
                        end

                        local ptr_ngram = ngram:data()
                        while true do
                           local k
                           if weight then
                              k = sampler()-1
                           else
                              k = math.random(list.ndoc)-1
                           end
                           -- get random line from that doc
                           local l = math.random(list.doc_size[k])-1
                           -- check whether that line is greater or equal than the required size
                           if list.line_size[k][l] >= size then
                              -- get random ngram from that line
                              local n = math.random(0,list.line_size[k][l]-size)
                             -- print(k,l,n,ngram:size())
                              ffi.copy(ptr_ngram,list.doc[k][l]+n,size*ffi.sizeof(c_int))
                              break
                           end
                        end
                        return ngram
                     end

                     function data:struct()
                        return list
                     end

                     function data:size()
                        return list.ndoc
                     end

                     function data:nbdocs()
                        return list.ndoc
                     end

                     function data:nblines()
                        return list.nline
                     end

                     function data:nbtokens()
                        return list.ntoken
                     end

                     function data:doclength(index)
                        return list.doc_size[index-1]
                     end

                     setmetatable(data,{
                           __index = function(self, index)
                                       return data:doc(index)
                                   end
                        })

                     return data
                 end
        })

function wikidocu.size(filename, threshold, verbose)

   local totallength = 0
   local ndoc=0
   local nline=0


   if verbose then print('# Sizing '..filename) end
   local files = scandir(filename)
   -- loop over files
   for f=1,#files do
      if verbose then print("# Reading "..files[f]) end
      local fstream = torch.DiskFile(files[f],'r'):binary()
      fstream:quiet()

      while not fstream:hasError() do
         -- skip doc id
         fstream:readInt()
         -- get doc number of lines
         local doclen = fstream:readInt()
         if not fstream:hasError() then
            local found=false
            for i=1,doclen do
               -- get number of tokens
               local linelen = fstream:readInt()
               fstream:seek(fstream:position()+linelen*ffi.sizeof(c_int) )
               if linelen > threshold then
                  totallength = totallength + linelen 
                  nline=nline+1
                  if not found then found=true end
               end
            end
            if found then ndoc=ndoc+1 end
         end
      end
      fstream:clearError()
      fstream:close()
   end
   if verbose then print("# docs= "..ndoc.. ", # lines= "..nline..", # tokens= "..totallength) end
   return ndoc,nline,totallength
end


function wikidocu.load(ptrfull_idx, ptrfull_line_size, ptrline, ptrdoc, ptrdoc_id, ptrdoc_size, ptrline_size, filename, threshold, verbose)

   local idx = torch.IntStorage()
   local idx_ptr = ffi.cast('THIntStorage*', torch.pointer(idx))
   idx_ptr.flag = 1
   
   if verbose then print('# Reading '..filename) end
   local files = scandir(filename)

   local idoc,iline,itoken = 0,0,0,0
   -- loop over files
   for f=1,#files do
      if verbose then  print("# Reading "..files[f]) end
      local fstream = torch.DiskFile(files[f],'r'):binary()
      fstream:quiet()

      while not fstream:hasError() do
         -- skip doc id
         local id=fstream:readInt()

         -- get doc number of lines
         local doclen = fstream:readInt()

         if not fstream:hasError() and doclen>0 then

            ptrline_size[idoc] = ptrfull_line_size+iline
            ptrdoc[idoc] = ptrline+iline

            local found=false
            local itr=0
            for i=1,doclen do
               -- get number of tokens
               local linelen = fstream:readInt()
               -- check whether the
               if linelen > threshold then
                  idx_ptr.size = linelen
                  idx_ptr.data = ptrfull_idx+itoken
                  ptrline[iline] = idx_ptr.data

                  -- read indices from file
                  fstream:readInt(idx)
                  -- store line size
                  ptrfull_line_size[iline] = linelen
                  -- increment line iterator
                  iline=iline+1
                  itr=itr+1
                  -- increment token iterator
                  itoken=itoken+linelen
                  if not found then found=true end
               else
                  -- skip this element
                  fstream:seek(fstream:position()+linelen*ffi.sizeof(c_int))
               end
            end
            if found then 
               ptrdoc_size[idoc]=itr
               ptrdoc_id[idoc]=id
               idoc=idoc+1
            end
         end
      end
      fstream:clearError()
      fstream:close()
   end
   if verbose then print("# done") end
end
