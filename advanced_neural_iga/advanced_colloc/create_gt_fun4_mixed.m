%% Create PDE model for fun_num=4: Mixed Pentagon (3 curved + 2 straight sides)
clear all
close all

% Parameters matching Python defaults
radius = 0.5;
center = [0.0, 0.0];
rotation = 0.0;
bulge = 0.18;
samples_per_side = 128;
curved_side_indices = [0, 2, 4];  % 0-indexed, sides 0, 2, 4 are curved

% Generate pentagon vertices
n_sides = 5;
angles = rotation + (2*pi / n_sides) * (0:n_sides-1);
vertices = [center(1) + radius * cos(angles);
            center(2) + radius * sin(angles)];

%% Build curved boundary using cubic Bezier curves for curved sides
% Curved sides: 0, 2, 4
% Straight sides: 1, 3

t_vals = linspace(0, 1, samples_per_side + 1)';  % Parameter values
boundary_points = [];

for i = 0:n_sides-1
    v0 = vertices(:, i+1);
    v1 = vertices(:, mod(i+1, n_sides) + 1);
    edge = v1 - v0;
    edge_len = norm(edge);
    
    if ismember(i, curved_side_indices)
        % Inward normal for CCW-ordered vertices
        n_in = [-edge(2), edge(1)]' / edge_len;
        curv = bulge * edge_len;
        
        % Cubic Bezier control points
        p0 = v0;
        p1 = (2/3) * v0 + (1/3) * v1 + curv * n_in;
        p2 = (1/3) * v0 + (2/3) * v1 + curv * n_in;
        p3 = v1;
        
        % Evaluate cubic Bezier curve
        one_minus_t = 1 - t_vals;
        b0 = one_minus_t.^3;
        b1 = 3 * one_minus_t.^2 .* t_vals;
        b2 = 3 * one_minus_t .* t_vals.^2;
        b3 = t_vals.^3;
        
        side_curve = b0 .* p0' + b1 .* p1' + b2 .* p2' + b3 .* p3';
    else
        % Straight line segment
        side_curve = v0' + t_vals * edge';
    end
    
    % Exclude the last point to avoid duplication at vertices
    boundary_points = [boundary_points; side_curve(1:end-1, :)];
end

% Remove nearly-duplicate points to avoid polyshape warnings
tolerance = 1e-10;
unique_points = boundary_points(1, :);
for i = 2:size(boundary_points, 1)
    dist_to_last = norm(boundary_points(i, :) - unique_points(end, :));
    if dist_to_last > tolerance
        unique_points = [unique_points; boundary_points(i, :)];
    end
end
boundary_points = unique_points;

% Create a polygon from the boundary points
polygon = polyshape(boundary_points(:,1), boundary_points(:,2));

% Visualize the geometry
figure('Name', 'Mixed Pentagon Geometry')
plot(polygon, 'FaceColor', [0.8, 0.9, 1.0], 'EdgeColor', 'blue', 'LineWidth', 2)
hold on
plot(vertices(1,:), vertices(2,:), 'ro', 'MarkerSize', 8, 'DisplayName', 'Vertices')
axis equal
grid on
title('Mixed Pentagon: 3 Curved (0,2,4) + 2 Straight Sides (1,3)')
xlabel('x')
ylabel('y')
legend
drawnow

%% Create PDE model using decomposed geometry format
model = createpde();

% Prepare boundary polygon for decomposed geometry matrix
pgon_x = boundary_points(:, 1);
pgon_y = boundary_points(:, 2);
n_boundary = length(pgon_x);

% Create decomposed geometry (gd) matrix in format for decsg
% Format: [type; n_vertices; x_coords; y_coords]
gd = [2; n_boundary; pgon_x; pgon_y];

% Create name string
ns = char('P1');
ns = ns';

% Set formula string
sf = 'P1';

% Create decomposed geometry using decsg
dl = decsg(gd, sf, ns);

% Create geometry from edges
geometryFromEdges(model, dl);

% Visualize geometry
figure
pdegplot(model, 'EdgeLabels', 'on')
axis equal
title('Mixed Pentagon: PDE Geometry')
drawnow

%% Specify PDE coefficients
fFun = @(location,state) ...
    sin(pi*location.y);
% Laplace equation: -div(grad(u)) = 0
specifyCoefficients(model, ...
    'm', 0, ...
    'd', 0, ...
    'c', 1, ...
    'a', 0, ...
    'f', fFun);

%% Apply boundary conditions
% Dirichlet: u = 0 on all boundaries
applyBoundaryCondition(model, 'dirichlet', 'Edge', 1:model.Geometry.NumEdges, 'u', 0);

%% Generate mesh
generateMesh(model, 'Hmax', 0.01);

figure
pdemesh(model)
axis equal
title('Mixed Pentagon: PDE Mesh')
numElements = size(model.Mesh.Elements, 2);
fprintf('Number of mesh elements: %d\n', numElements);
fprintf('Number of mesh nodes: %d\n', size(model.Mesh.Nodes, 2));
drawnow

%% Solve PDE (Laplace equation with homogeneous Dirichlet BC)
fprintf('Solving PDE...\n');
result = solvepde(model);

u = result.NodalSolution;

% Visualize solution
figure
pdeplot(model, 'XYData', u, 'Contour', 'on', 'ColorMap', 'hot')
axis equal
colorbar
title('Solution to Laplace Equation on Mixed Pentagon')
fprintf('PDE solved successfully\n');
drawnow

%% Sample random points inside the domain and interpolate solution
N = 10000;

% Get bounding box from boundary points
xmin = min(boundary_points(:,1));
xmax = max(boundary_points(:,1));
ymin = min(boundary_points(:,2));
ymax = max(boundary_points(:,2));

fprintf('\nSampling %d points inside domain...\n', N);
fprintf('Bounding box: [%.4f, %.4f] x [%.4f, %.4f]\n', xmin, xmax, ymin, ymax);

xrand = [];
yrand = [];
count = 0;

while count < N
    % Generate a batch of candidate points
    n_missing = N - count;
    n_candidates = max(2048, n_missing * 3);
    
    xc = xmin + (xmax - xmin) * rand(n_candidates, 1);
    yc = ymin + (ymax - ymin) * rand(n_candidates, 1);
    
    % Check if points are inside the polygon
    inside = isinterior(polygon, xc, yc);
    
    % Take as many as needed
    nnew = min(sum(inside), n_missing);
    
    if nnew > 0
        idx = find(inside, nnew);
        xrand = [xrand; xc(idx)];
        yrand = [yrand; yc(idx)];
        count = count + nnew;
    end
    
    if mod(count, 1000) == 0
        fprintf('  Sampled %d points...\n', count);
    end
end

% Trim to exact size if needed
xrand = xrand(1:N);
yrand = yrand(1:N);

fprintf('Total sampled: %d points\n', length(xrand));

%% Interpolate solution at sampled points
fprintf('Interpolating solution at sampled points...\n');
uq = interpolateSolution(result, xrand, yrand);

%% Prepare and save data
data = [double(xrand), double(yrand), double(uq)];

output_filename = 'poisson_samples_fun4_mixed_pentagon.csv';
fprintf('Saving to: %s\n', output_filename);

fid = fopen(output_filename, 'w');
fprintf(fid, 'x,y,u\n');

for k = 1:size(data, 1)
    fprintf(fid, '%.17g,%.17g,%.17g\n', data(k,1), data(k,2), data(k,3));
end

fclose(fid);

fprintf('Done! Data saved to %s\n', output_filename);
fprintf('Data dimensions: %d samples x 3 columns\n', size(data, 1));

% Summary
fprintf('\n========== SUMMARY ==========\n');
fprintf('Geometry: Mixed Pentagon (fun_num=4)\n');
fprintf('  - Radius: %.4f\n', radius);
fprintf('  - Center: (%.4f, %.4f)\n', center(1), center(2));
fprintf('  - Rotation: %.4f rad\n', rotation);
fprintf('  - Bulge (curvature): %.4f\n', bulge);
fprintf('  - Curved sides: 0, 2, 4\n');
fprintf('  - Straight sides: 1, 3\n');
fprintf('  - Samples per side: %d\n', samples_per_side);
fprintf('PDE: Laplace equation, u=0 on boundary\n');
fprintf('Mesh: %d elements, %d nodes\n', numElements, size(model.Mesh.Nodes, 2));
fprintf('Data: %d sample points\n', N);
fprintf('Output file: %s\n', output_filename);
fprintf('=============================\n');
