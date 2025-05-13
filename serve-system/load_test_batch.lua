wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
local B = 8
local single = '{"candidate":[[0.1,0.2,0.3]],"history":[[[0.1,0.2,0.3],[0.4,0.5,0.6]]],"length":[2]}'
local arr = {}
for i=1,B do table.insert(arr, single) end
local body = string.format('{"model":"din","batch":[%s]}', table.concat(arr, ","))
function request() return wrk.format(nil, "/infer_batch/", nil, body) end