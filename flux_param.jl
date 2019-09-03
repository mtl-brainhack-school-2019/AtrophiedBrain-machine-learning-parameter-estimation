using Flux, DiffEqFlux, DifferentialEquations, Plots, StatsBase
using Flux: @epochs
using Base.Iterators: repeated, partition

const rc = 3
const u0 = Float32[3.78; 3.42; 2.44]
const datasize = 178
const tspan = (1.0f0,90.0f0)
const num_train_sets = 1000
const num_test_sets = 100

function f2(du, u, p, t)
    #a, b = p
    a = p

    @inbounds for i in 1:rc
        #t2 = 0.0
        #@inbounds for j in 1:rc
        #    if j != i
        #        t2 += u[j]
        #    end
        #end
        du[i] = u[i]*a #+ t2*b
    end
end

function random_u0(num)
    set = collect(2.84:0.02:4.22)
    k = zeros(num)
    sample!(set, k)
end

function random_alpha(num)
    set1 = collect(-0.05:0.005:-0.005)
    #set = vcat(set1, collect(0.005:0.005:0.05))
    k = zeros(num)
    sample!(set1, k)
end

# function random_beta(num)
#     set = collect(0.0005:0.0005:0.005)
#     k = zeros(num)
#     sample!(set, k)
# end

const ps = -0.05 #(-0.05, 0.003)
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(f2,u0,tspan, ps)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

Xs = [ode_data[1,:], ode_data[2, :], ode_data[3, :]]
Ys = [[-0.05], [-0.05], [-0.05]] #[[-0.05, 0.003],[-0.05, 0.003],[-0.05, 0.003]]

Xss_train = Vector{Float32}[]
Yss_train = Vector{Float64}[]
for i in 1:num_train_sets
    global Xss_train, Yss_train
    a = random_alpha(1)[1]
    #b = random_beta(1)[1]
    u0 = random_u0(rc)

    t = range(tspan[1],tspan[2],length=datasize)
    prob = ODEProblem(f2,u0,tspan, a)
    ode_data = Array(solve(prob,Tsit5(),saveat=t))

    push!(Xss_train,ode_data[1, :])
    push!(Yss_train,[a])
end

Xss_test = Vector{Float32}[]
Yss_test = Vector{Float64}[]
for i in 1:num_test_sets
    global Xss_test, Yss_test
    a = random_alpha(1)[1]
    #b = random_beta(1)[1]
    u0 = random_u0(rc)

    t = range(tspan[1],tspan[2],length=datasize)
    prob = ODEProblem(f2,u0,tspan, a)
    ode_data = Array(solve(prob,Tsit5(),saveat=t))

    push!(Xss_test,ode_data[1, :])
    push!(Yss_test,[a])
end
#
# train_notused = [(Xss[i], Yss[i])
#          for i in partition(1:num_train_sets, floor(Int,(num_train_sets)/10))]

m = Chain(
  Dense(datasize,datasize),
  LSTM(datasize, 256),
  LSTM(256, 128),
  Dense(128, 1, tanh))

function loss(xs, ys)
    preds = m(xs)
    l = sum(Flux.mse.(preds, ys))
    Flux.truncate!(m)
    return l
end

opt = ADAM(0.01)
tx, ty = (Xss_train[2], Yss_train[2])
evalcb = () -> @show loss(tx, ty)

@epochs 10 Flux.train!(loss, params(m), zip(Xss_train, Yss_train), opt,
            cb = Flux.throttle(evalcb, 500))

println(sum(Flux.mse.(m(Xss_test[1]), Yss_test[1])))
