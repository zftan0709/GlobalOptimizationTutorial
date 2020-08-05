% build a random grid graph
d = 2; % 2D planar graph
nrNodes = 100; % graph has n = 100 vertices
probLC = 0.8; % probability of loop closures in the grid graph
rotStd = 0.01; % noise standard deviation for rotation
tranStd = 0.01; % noise standard deviation for translation
graph = grid_random_graph_2D(nrNodes, ...
 'RotationStd', rotStd, ...
 'TranslationStd', tranStd, ...
 'Scale', 1.0, ...
 'LoopClosureProbability', probLC);
nrEdges = size(graph.edges,1);
% save the ground-truth rotations
R_gt = zeros(d*nrNodes,d);
for i=1:nrNodes
 R_gt(blkIndices(i,d),:) = rot2D(graph.poses_gt(i,3));
end
if norm(R_gt(1:2,1:2,1)-eye(2)) > 1e-6
 error('first rotation != identity')
end
fprintf(['Random 2D grid graph: number of nodes: %d, ' ...
 'number of edges: %d.\n'], ...
 nrNodes, nrEdges);

% build the data matrix R_tilde (d=2 for 2D planar graph)
R_tilde = build_R_tilde(graph.edges,nrNodes,d);
spy(R_tilde);
ylim([0, 50]), xlim([0,50]); % only visualize part of the sparsity pattern
title('Sparsity pattern of $\tilde{R}$',...
 'Interpreter','latex','fontsize',20)

% implement the SDP in problem (9)
t_start = tic;
cvx_quiet(true)
cvx_begin sdp % start CVX in SDP mode
cvx_precision best % ask for the best precision
cvx_solver sedumi % choose solver sedumi (sdpt3, mosek et al.)
% define the decision variable Z
variable Z(d*nrNodes, d*nrNodes) symmetric
% define the objective function (we include the constant term in Eq.(3) as well)
f_cost = trace(R_tilde * Z) + 2 * d * nrEdges;
minimize(f_cost)
% define the constraints of the SDP
subject to
Z >= 0 % the semidefinite constraint
for i = 1:nrNodes
 Z(blkIndices(i,d),blkIndices(i,d)) == eye(2)
end
% then solve the SDP!
cvx_end
t_sdp = toc(t_start);

% extract SDP solutions and check relaxation tightness
f_sdp = cvx_optval;
Z_star = full(Z);
rank_Z = rank(Z_star, 1e-3);
isExact = (rank_Z == d);
fprintf(['f_sdp = %g, rank_Z = %d, solver time = %g[s], ' ...
 'relaxation tightness: %s.\n'], ...
 f_sdp, rank_Z, t_sdp, string(isExact));

figure;
bar(eig(Z_star));
title('Eigenvalues of $Z^\star$','Interpreter','latex','FontSize',20)

% Extract solution to the original problem from SDP solution
R_est = zeros(d*nrNodes,d);
for i = 1:nrNodes
 R_est(blkIndices(i,d),:) = project2SO2( Z_star(1:d, blkIndices(i,d))' );
end
% evaluate the objective function on R_est to get f_est
f_est = trace(R_tilde * R_est * R_est') + 2 * d * nrEdges;
relative_duality_gap = abs(f_est - f_sdp) / f_est;
fprintf(['SDP relaxation: f_sdp = %g, f_est = %g, ' ...
 'relative duality gap = %g.\n'],...
 f_sdp,f_est,relative_duality_gap);

R_err = rad2deg( get_rot_error(R_est,R_gt,d) );
min_R_err = min(R_err);
max_R_err = max(R_err);
mean_R_err = mean(R_err);
fprintf(['SDP estimation: min R_err = %g[deg], ' ...
    'max R_err = %g[deg], mean R_err = %g[deg].\n'],...
 min_R_err,max_R_err,mean_R_err);
