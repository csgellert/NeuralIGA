%% Create PDE model
clear all
model = createpde();

%% Define pentagon vertices
xv = 1*[1 cos(1*2*pi/5) cos(2*2*pi/5) cos(3*2*pi/5) cos(4*2*pi/5)];
yv = 1*[0 sin(1*2*pi/5) sin(2*2*pi/5) sin(3*2*pi/5) sin(4*2*pi/5)];
%%
fFun = @(location,state) ...
    sin(pi*location.y);
fFun2 = @(location,state) ...
    1^2-(location.x^2+location.y^2);

%% Geometry description matrix
gd = [2;          % polygon
      5;          % number of vertices
      xv(:);
      yv(:)];

ns = char('P1');
ns = ns';

sf = 'P1';

dl = decsg(gd,sf,ns);

geometryFromEdges(model,dl);

figure
pdegplot(model,'EdgeLabels','on')
axis equal

%%
specifyCoefficients(model,...
    'm',0,...
    'd',0,...
    'c',1,...
    'a',0,...
    'f',fFun);

applyBoundaryCondition(model,...
    'dirichlet',...
    'Edge',[1,5],...
    'u',0);
applyBoundaryCondition(model,...
    'dirichlet',...
    'Edge',[2,3,4],...
    'u',fFun2);%0

generateMesh(model,'Hmax',0.001);
figure
pdemesh(model)
axis equal
title('PDE Toolbox Mesh')
%% solve
result = solvepde(model);

u = result.NodalSolution;

figure
pdeplot(model,'XYData',u,'Contour','on')
numElements = size(model.Mesh.Elements, 2);
disp(numElements)
%%
% Use the same polygon that defines the PDE model so sampled points stay inside.
pg = polyshape(xv,yv);

N = 10000;

% Bounding box
xmin = min(xv);
xmax = max(xv);
ymin = min(yv);
ymax = max(yv);

xrand = zeros(N,1);
yrand = zeros(N,1);

count = 0;

while count < N

    % Generate a batch of candidates
    M = max(2*(N-count),1000);

    xc = xmin + (xmax-xmin)*rand(M,1);
    yc = ymin + (ymax-ymin)*rand(M,1);

    inside = isinterior(pg,xc,yc);

    nnew = min(sum(inside),N-count);

    idx = find(inside,nnew);

    xrand(count+1:count+nnew) = xc(idx);
    yrand(count+1:count+nnew) = yc(idx);

    count = count + nnew;
end
%% interpolate solutions

uq = interpolateSolution(result,xrand,yrand);

%% save
data = [double(xrand), double(yrand), double(uq)];

fid = fopen('poisson_samples.csv','w');

fprintf(fid,'x,y,u\n');

for k = 1:size(data,1)
    fprintf(fid,'%.17g,%.17g,%.17g\n', ...
        data(k,1), data(k,2), data(k,3));
end

fclose(fid);