wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"model":"din","data":{"candidate":[[0.1,0.2,0.3]],"history":[[[0.1,0.2,0.3],[0.4,0.5,0.6]]],"length":[2]}}'
function request() return wrk.format(nil, "/infer/", nil, wrk.body) end