using Base: Float64
using POMDPModelTools:stateindex
using AutoViz
using BeliefUpdaters
using Distributions
using FIB
using GridInterpolations
using LinearAlgebra
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using POMDPSolve
using POMDPModels # this defines TigerPOMDP
using POMDPs
using POMDPModelTools
# using POMCPOW
using QMDP
using Reel
using Random
using SARSOP
using StatsBase
using Test

########################################
# Type definitions
########################################
mutable struct Pose3D
    x::Float64
    y::Float64
    θ::Float64
end
Pose3D() = Pose3D(0., 0., 0.)
function Base.:(*)(p::Pose3D, w::Float64)
    return Pose3D(p.x * w, p.y * w, p.θ * w)
end
function Base.:(==)(a::Pose3D, b::Pose3D)
    return a.x == b.x && a.y == b.y && a.θ == b.θ
end
function Base.:(+)(a::Pose3D, b::Pose3D)
    Pose3D(a.x + b.x, a.y + b.y, a.θ + b.θ)
end
function Base.hash(p::Pose3D, h::UInt64=zero(UInt64))
    return hash(p.x, hash(p.y, hash(p.θ, h)))
end

mutable struct Environment
    speed_limit::Float64
end

struct VehicleState
    posG::Pose3D
    v::Float64
end
function Base.:(+)(a::VehicleState, b::VehicleState)
    VehicleState(a.posG + b.posG, a.v + b.v)
end
function Base.hash(s::VehicleState, h::UInt64=zero(UInt64))
    return hash(s.v, hash(s.posG, h))
end
VehicleState() = VehicleState(Pose3D(), 0.)
VehicleState(x0::Float64, v0::Float64) = VehicleState(Pose3D(x0, 0., 0.), v0)
VehicleState(v0::Float64) = VehicleState(Pose3D(), v0)
function Base.:(*)(s::VehicleState, w::Float64)
    return VehicleState(s.posG * w, s.v * w)
end

mutable struct AccState
    crash::Bool
    ego::VehicleState
    car::VehicleState
end
# AccState(crash0::Bool) = AccState(crash0, VehicleState(15.),
#                                   VehicleState(10.))
function Base.:(*)(s::AccState, w::Float64)
    return AccState(s.crash, s.ego * w, s.car * w)
end
function Base.:(+)(a::AccState, b::AccState)
    AccState(a.crash | b.crash, a.ego + b.ego, a.car + b.car)
end
function Base.copyto!(a::AccState, b::AccState)
    a.crash = b.crash
    a.ego = b.ego
    a.car = b.car
end
function Base.isless(a::AccState, b::AccState)
    return false
end
function Base.isless(a::AccState, w::Float64)
    return false
end
function Base.hash(s::AccState, h::UInt64=zero(UInt64))
    return hash(s.crash, hash(s.ego, hash(s.car, h)))
end
function Base.zero(::Type{AccState})
    return AccState(false, VehicleState(0., 0., ), VehicleState(1000., 0.))
end
function Base.:(==)(a::AccState, b::AccState)
    return a.crash == b.crash && a.ego == b.ego && a.car == b.car
end

const AccObs = AccState
mutable struct AccAction
    acc::Float64
end
function Base.copyto!(a::AccAction, b::AccAction)
    a.acc = b.acc
end
function Base.hash(a::AccAction, b::AccAction)
    return a.acc == b.acc
end
function Base.:(==)(a::AccAction, b::AccAction)
    return a.acc = b.acc
end

########################################
# Model
########################################
mutable struct AccPOMDP <: POMDP{AccState,AccAction,AccObs}
    max_acc::Float64
    max_dec::Float64
    pos_res::Float64
    vel_res::Float64
    pose_start::Pose3D
    pose_end::Pose3D
    road_end::Pose3D
    ΔT::Float64
    a_noise::Float64
    pos_obs_noise::Float64
    vel_obs_noise::Float64
    collision_cost::Float64
    action_cost::Float64
    goal_reward::Float64
    γ::Float64
    env::Environment
    size_x::Int64
    size_v::Int64
    grid::AbstractGrid
end

function CreateAccPOMDP(;
    max_acc::Float64=2.,
    max_dec::Float64=4.,
    pos_res::Float64=1.,
    vel_res::Float64=1.,
    pose_start::Pose3D=Pose3D(0., 0., 0.),
    pose_end::Pose3D=Pose3D(5., 0., 0.),
    road_end::Pose3D=Pose3D(10., 0., 0.),
    ΔT::Float64=0.5,
    a_noise::Float64=0.5,
    pos_obs_noise::Float64=0.5,
    vel_obs_noise::Float64=0.5,
    collision_cost::Float64=-1.,
    action_cost::Float64=0.0,
    goal_reward::Float64=2.,
    γ::Float64=0.95,
    speed_limit::Float64=3.)
    size_x = Int(floor(road_end.x / pos_res) + 1)
    size_v = Int(floor(speed_limit / vel_res) + 1)
    rect = RectangleGrid(LinRange(0., road_end.x, size_x),
                         LinRange(0., speed_limit, size_v))
    # @show rect
    return AccPOMDP(max_acc, max_dec, pos_res, vel_res, pose_start,
                    pose_end, road_end, ΔT, a_noise, pos_obs_noise,
                    vel_obs_noise, collision_cost, action_cost,
                    goal_reward, γ, Environment(speed_limit), size_x, size_v,
                    rect)
end

########################################
# Reward
########################################
function POMDPs.reward(pomdp::AccPOMDP, ::AccState, a::AccAction, sp::AccState)
    r = 0.
    if sp.crash
        r += pomdp.collision_cost
    end
    if sp.ego.posG.x >= pomdp.pose_end.x
        r += pomdp.goal_reward
    elseif a.acc > 0.
        r += pomdp.action_cost
    else
        r += pomdp.action_cost
    end
    return r
end
function POMDPs.reward(pomdp::AccPOMDP, s::AccState, a::AccAction)
    return reward(pomdp, s, a, s)
end

########################################
# Miscellaneous
########################################
function POMDPs.isterminal(pomdp::AccPOMDP, s::AccState)
    return s.crash || s.ego.posG.x >= pomdp.pose_end.x
end
function POMDPs.discount(pomdp::AccPOMDP)
    return pomdp.γ
end
function is_crash(ego::VehicleState, car::VehicleState)
    # Will move vehicle lenght, width into parameters
    return ego.posG.x + 1.0 >= car.posG.x
end
function Base.show(io::IO, s::AccState)
    print(io, "AccState: ego(", s.ego.posG.x, ", ", s.ego.v, "), car(",
          s.car.posG.x, ", ", s.car.v, ")")
end

########################################
# States
########################################
function POMDPs.states(pomdp::AccPOMDP)
    X = LinRange(0., pomdp.road_end.x, pomdp.size_x)
    V = LinRange(0., pomdp.env.speed_limit, pomdp.size_v)
    state_space = Vector{AccState}()
    for x0 in X
        for v0 in V
            for x1 in X
                for v1 in V
                    ego = VehicleState(x0, v0)
                    car = VehicleState(x1, v1)
                    push!(state_space, AccState(is_crash(ego, car), ego, car))
                end
            end
        end
    end
    return state_space
end

function POMDPs.stateindex(pomdp::AccPOMDP, s::AccState)
    x_ego = s.ego.posG.x
    v_ego = s.ego.v
    x_car = s.car.posG.x
    v_car = s.car.v
    x_ego_ind = Int(ceil((x_ego - 0.) / pomdp.pos_res)) + 1
    v_ego_ind = Int(ceil(v_ego / pomdp.vel_res)) + 1
    x_car_ind = Int(ceil((x_car - 0.) / pomdp.pos_res)) + 1
    v_car_ind = Int(ceil(v_car / pomdp.vel_res)) + 1
    ind = LinearIndices((pomdp.size_x, pomdp.size_v, pomdp.size_x, pomdp.size_v))[x_ego_ind, v_ego_ind, x_car_ind, v_car_ind]
    return ind
end

function POMDPs.convert_s(::Type{S}, v::AbstractArray, pomdp::AccPOMDP) where (S <: AccState)
    id = first(v)
    x_ego_ind, v_ego_ind, x_car_ind, v_car_ind = CartesianIndices((pomdp.size_x, pomdp.size_v, pomdp.size_x, pomdp.size_v))[id].I
    x_ego = Float64(floor(x_ego_ind - 1))
    v_ego = Float64(floor(v_ego_ind - 1))
    x_car = Float64(floor(x_car_ind - 1))
    v_car = Float64(floor(v_car_ind - 1))
    ego = VehicleState(x_ego, v_ego)
    car = VehicleState(x_car, v_car)
    return AccState(is_crash(ego, car), ego, car)
end

########################################
# Actions
########################################
# POMDPs.actions(::AccPOMDP) = [AccAction(-4.0), AccAction(-3.0),
#                               AccAction(-2.0), AccAction(-1.0),
#                               AccAction(0.), AccAction(1.0),
#                               AccAction(2.0)]
POMDPs.actions(::AccPOMDP) = [AccAction(-2.0),
                              AccAction(0.),
                              AccAction(2.0)]
function POMDPs.actionindex(::AccPOMDP, a::AccAction)
    if a.acc == -2.
        return 1
    elseif a.acc == 0.
        return 2
    elseif a.acc == 2.
        return 3
    else
        @assert false "unsupported action: $(a.acc)"
    end
    # if a.acc == -4.
    #     return 1
    # elseif a.acc == -3.
    #     return 2
    # elseif a.acc == -2.
    #     return 3
    # elseif a.acc == -1.
    #     return 4
    # elseif a.acc == 0.
    #     return 5
    # elseif a.acc == 1.
    #     return 6
    # elseif a.acc == 2.
    #     return 7
    # else
    #     @assert 0 "unsupported action: $(a.acc)"
    # end
end

########################################
# Observations
########################################
function POMDPs.observations(pomdp::AccPOMDP)
    return states(pomdp)
end
function POMDPs.obsindex(pomdp::AccPOMDP, s::AccState)
    return stateindex(pomdp, s)
end

########################################
# Initialstate
########################################
function POMDPs.initialstate(pomdp::AccPOMDP)
    env = pomdp.env
    V = LinRange(0, env.speed_limit, pomdp.size_v)
    X = LinRange(pomdp.pose_start.x, pomdp.pose_end.x, pomdp.size_x)
    states = AccState[]
    ego = VehicleState(pomdp.pose_start, 0.)
    for x in X
        for v in V
            car = VehicleState(x, v)
            push!(states, AccState(is_crash(ego, car), ego, car))
        end
    end
    probs = ones(length(states))
    # normalize!(probs, 1)
    return SparseCat(states, probs)
end

########################################
# Transition
########################################
function POMDPs.transition(pomdp::AccPOMDP, s::AccState, a::AccAction)
    dt = pomdp.ΔT
    ego_states, ego_probs = ego_transition(pomdp, s.ego, a, dt)
    car_states, car_probs = car_transition(pomdp, s.car, dt)
    
    n_next_states = length(ego_states) * length(car_states)
    
    next_states = Vector{AccState}(undef, n_next_states)
    next_probs = zeros(n_next_states)
    ind = 1
    for (i, ego) in enumerate(ego_states)
        for (j, car) in enumerate(car_states)
            crash = is_crash(ego, car)
            next_states[ind] = AccState(crash, ego, car)
            next_probs[ind] = car_probs[j] * ego_probs[i]
            ind += 1
        end
    end
    normalize!(next_probs, 1)
    SparseCat(next_states, next_probs)
end
function ego_transition(pomdp::AccPOMDP, ego::VehicleState, a::AccAction,
    dt::Float64)
    x = ego.posG.x + ego.v * dt + 0.5 * a.acc * dt^2
    if x <= ego.posG.x
        x = ego.posG.x
    end
    v = ego.v + a.acc * dt
    if v <= 0.0
        v = 0.0
    end
    index, weight = interpolants(pomdp.grid, [x, v])
    n_pts = length(index)
    states = Array{VehicleState}(undef, n_pts)
    probs = Array{Float64}(undef, n_pts)
    for i = 1:n_pts
        xg, vg = ind2x(pomdp.grid, index[i])
        states[i] = VehicleState(xg, vg)
        probs[i] = weight[i]
    end
    # normalize!(probs, 1)
    return states, probs
end
function car_transition(pomdp::AccPOMDP, car::VehicleState,
    dt::Float64)
    x = car.posG.x + car.v * dt
    if x > pomdp.road_end.x
        return [VehicleState(pomdp.road_end.x, car.v)], [1.]
    end
    states = VehicleState[]
    probs = Float64[]
    for a_noise in [-pomdp.a_noise, 0., pomdp.a_noise]
        x += 0.5 * a_noise * dt^2
        v = car.v + a_noise * dt
        ind, weight = interpolants(pomdp.grid, [x, v])
        for i in 1:length(ind)
            xg, vg = ind2x(pomdp.grid, ind[i])
            car_state = VehicleState(xg, vg)
            if !(car_state in states)
                push!(states, car_state)
                push!(probs, weight[i])
            else
                state_ind = findall(x -> x == car_state, states)
                probs[state_ind] += weight[i]
            end
        end
    end
    normalize!(probs, 1)
    # Why this?
    probs = probs .+ maximum(probs)
    normalize!(probs)
    return states, probs
end

########################################
# Helper
########################################
function off_road(pomdp::AccPOMDP, car::VehicleState)
    return car.posG.x > pomdp.road_end.x
end
function in_bounds(pomdp::AccPOMDP, car::VehicleState)
    return car.posG.x >= 0. && car.posG.x <= pomdp.road_end.x &&
        car.v >= 0. && car.v <= pomdp.env.speed_limit
end

########################################
# Observation
########################################
function POMDPs.observation(pomdp::AccPOMDP, sp::AccState)
    if off_road(pomdp, sp.car)
        o = AccObs(false, sp.ego, VehicleState(pomdp.road_end.x, sp.car.v))
        return SparseCat([o], [1.])
    elseif is_crash(sp.ego, sp.car)
        return SparseCat([sp], [1.])
    end
    ego = sp.ego
    car = sp.car
    neighbors = Vector{VehicleState}(undef, 9)
    neighbors[1] = VehicleState(car.posG.x - pomdp.pos_res, car.v)
    neighbors[2] = VehicleState(car.posG.x, car.v)
    neighbors[3] = VehicleState(car.posG.x + pomdp.pos_res, car.v)
    neighbors[4] = VehicleState(car.posG.x - pomdp.pos_res, car.v - pomdp.vel_res)
    neighbors[5] = VehicleState(car.posG.x, car.v)
    neighbors[6] = VehicleState(car.posG.x + pomdp.pos_res, car.v - pomdp.vel_res)
    neighbors[7] = VehicleState(car.posG.x - pomdp.pos_res, car.v + pomdp.vel_res)
    neighbors[8] = VehicleState(car.posG.x, car.v)
    neighbors[9] = VehicleState(car.posG.x + pomdp.pos_res, car.v + pomdp.vel_res)
    
    obss = AccObs[]
    sizehint!(obss, 9)
    for neighbor in neighbors
        if in_bounds(pomdp, neighbor) && !is_crash(ego, neighbor)
            push!(obss, AccObs(false, ego, neighbor))
        end
    end
    probs = ones(length(obss))
    normalize!(probs, 1)
    return SparseCat(obss, probs)
end
function POMDPs.observation(pomdp::AccPOMDP, ::AccAction, sp::AccState)
    observation(pomdp, sp)
end
function POMDPs.observation(pomdp::AccPOMDP, ::AccState, a::AccState,
    sp::AccState)
    observation(pomdp, a, sp)
end

########################################
# Initialobs
########################################
initialobs(pomdp::AccPOMDP, s::AccState) =
    observation(pomdp, s)

########################################
# Convert s
########################################
function POMDPs.convert_s(::Type{A}, s::AccState, pomdp::AccPOMDP) where A <: AbstractArray
    return convert(A, [stateindex(pomdp, s)])
end

########################################
# Belief and updater
########################################
const AccBelief = SparseCat
mutable struct AccUpdater <: Updater
    pomdp::AccPOMDP
end
POMDPs.initialize_belief(::AccUpdater, d::Any) = d
function POMDPs.update(up::AccUpdater, b::AccBelief,
    a::AccAction, o::AccObs)
    bp = AccBelief()
    pomdp = b.pomdp
    pomdp_states = ordered_states(pomdp)

    for (_, sp) in enumerate(pomdp_states)
        od = observation(pomdp, a, sp)
        prob_o = pdf(od, o)
        if prob_o == 0.0
            continue
        end
        b_sum = 0.0
        for (j, s) in enumerate(b.vals)
            td = transition(pomdp, s, a)
            prob_sp = pdf(td, sp)
            @inbounds b_sum += prob_sp * b.probs[j]
        end
        if b_sum != 0.
            push!(bp.vals, sp)
            push!(bp.probs, prob_o * b_sum)
        end
    end

    if sum(bp.probs) == 0.0
        println("Invalid update for: ", b, " ", a, " ", o)
        throw("UpdateError")
    else
        normalize!(bp.probs, 1)
    end
    return bp
end

@testset  "Acc POMDP" begin
    rng = MersenneTwister(1)

    pomdp = CreateAccPOMDP()
    pomdp_states = ordered_states(pomdp)
    @show length(pomdp_states)

    if true
        # Solve
        solver = POMDPSolveSolver()
        # solver = QMDPSolver()
        # solver = FIBSolver()
        # solver = SARSOPSolver()
        policy0 = solve(solver, pomdp) # compute a pomdp policy

        # rand_policy = RandomPolicy(pomdp)
        # rollout_sim = RolloutSimulator(max_steps=10)
        # rand_reward = simulate(rollout_sim, pomdp, rand_policy)
        # @show rand_reward
    else

        # still_policy = FunctionPolicy(s -> AccAction(0.)) # no acceleration
        # policy = RandomPolicy(pomdp, rng=rng)
        # up = NothingUpdater() # no belief update, to use the observation as
        # # input to the policy see PreviousObservationUpdater
        # # from BeliefUpdaters.jl
        # s0 = initialstate(pomdp, rng) # generate an initial state
        # # set up the simulation
        # hr = HistoryRecorder(rng=rng, max_steps=100)
        # @time hist = simulate(hr, pomdp, policy, up, nothing, s0);
        # state_history = state_hist(hist)
        # x = [s.ego.posG.x for s in state_history]
        # v = [s.ego.v for s in state_history]
        # @show x
        # @show v
        # # plot(x, v)
        # # plot(state_history)
        # @show length(state_history)
        # return


        # Simulation
        policy = RandomPolicy(pomdp, rng=rng)
        up = KMarkovUpdater(5)
        s0 = rand(rng, initialstate(pomdp))
        # s0 = AccState(false, VehicleState(0., 20.), VehicleState(20., 15.))
        # @show s0
        initial_observation = rand(rng, initialobs(pomdp, s0))
        initial_obs_vec = fill(initial_observation, 5)
        hr = HistoryRecorder(rng=rng, max_steps=100)
        @time hist = simulate(hr, pomdp, policy, up, initial_obs_vec, s0)
        state_history = state_hist(hist)
        x_car = [s.car.posG.x for s in state_history]
        v_car = [s.car.v for s in state_history]
        x_ego = [s.ego.posG.x for s in state_history]
        v_ego = [s.ego.v for s in state_history]
        dx = [s.car.posG.x - s.ego.posG.x for s in state_history]
        # @show x_car
        # @show v_car
        # @show x_ego
        # @show v_ego
        # @show dx

        # @show hist
        # @show n_steps(hist)
        @test n_steps(hist) > 1
    end


end
