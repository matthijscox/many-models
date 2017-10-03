
% readtable doesn't like url's
T = readtable('many_models_data.csv'); 

% trying urlread instead
% tu = urlread('https://raw.githubusercontent.com/matthijscox/many-models/master/many_models_data.csv');
% tu = regexp(tu,',|\n','split'); % split by comma's or new lines
% tu = regexprep(tu,'\s',''); % remove whitespace
% tu = reshape(tu(1:end-1),5,[])'; % reshape
% tu(2:end,3:end) = cellfun(@str2double,tu(2:end,3:end),'unif',0);
% T = cell2table(tu(2:end,:));
% T.Properties.VariableNames = tu(1,:);

%% using matlab's functions (using fit = 35s, using matrices = 0.9s)
 tic
 [G,num_G,cat_G] = findgroups(T.num_group,T.cat_group);
 Y = splitapply(@calc_poly_residuals_with_matrices_sa,T.A,T.B,T.Z,G);
 toc

%% array based approach (2.3s using find, 0.7s using find_in_sorted)

tic
[G,num_G,cat_G] = findgroups(T.num_group,T.cat_group); % ~0.2 s
ngroups = numel(num_G);

A = T.A;
B = T.B;
Z = T.Z;

resids = zeros(numel(Z),1);
for n=1:ngroups
    %group_idx = find(G==n); % this is the slowest part
    group_idx = find_in_sorted(G,n); % faster
    x = A(group_idx);
    y = B(group_idx);
    z = Z(group_idx);
    resids(group_idx) = calc_poly_residuals_with_matrices(x,y,z);
end
toc

sqrt(sum(resids.^2)/numel(resids))-0.878223092545709


function resids=calc_poly_residuals(x,y,z)
    [~,~,output]=fit([x, y],z,'poly33');
    resids = {output.residuals};
end

function resids=calc_poly_residuals_with_matrices_sa(x,y,z)
    resids = {calc_poly_residuals_with_matrices(x,y,z)};
end

function resids=calc_poly_residuals_with_matrices(x,y,z)
    M = model_matrix_poly(x,y,3); 
    c = M\z; 
    resids = z - M*c;
end

function M=model_matrix_poly(x,y,poly_order)
    ndat=numel(x);
    npoly=(poly_order+1)*(poly_order+2);
    M=ones(ndat,npoly/2);
    for pow=1:poly_order
        for pow_y=0:pow
            pow_x=pow-pow_y;
            M(:,pow*(pow+1)/2+pow_y+1)=(x.^pow_x.*y.^pow_y);
        end
    end
end

function idx=find_in_sorted(x,searchfor)
% Adapted from https://stackoverflow.com/questions/20166847/faster-version-of-find-for-sorted-vectors-matlab
    nx = numel(x);
    a=1;
    b=nx;
    c=1;
    d=nx;
    while (a+1<b||c+1<d)
        lw=(floor((a+b)/2));
        if (x(lw)<searchfor)
            a=lw;
        else
            b=lw;
        end
        lw=(floor((c+d)/2));
        if (x(lw)<=searchfor)
            c=lw;
        else
            d=lw;
        end
    end
    if b==2 && x(1)==searchfor
        b=1; %edge case
    end
    if c==nx-1 && x(nx)==searchfor
        c=nx; %edge case
    end
    idx = (b:c)';
end