function unew = Vcycle(u,f,nsmooth,h)
    if(size(u,1) == 1)
        unew = f;
        return;
    end

    unew = u;

    % Smooting solution
    for i=1:nsmooth
        unew = jacobi(unew,f,h);
    end

    % Fining residual
    r = -laplacian(unew,f,h)+f;

    % Restricting residual to coarse grid
    rc = restrict(r);

    % Initial guess on error
    ec = zeros(size(rc));

    % Recursion
    ec = Vcycle(ec,rc,nsmooth,h);

    % Interpolating error
    ef = interpolate(ec);

    % Updating solution
    unew = unew + ef;

    % Smoothing solution
    for i=1:nsmooth
        unew = jacobi(unew,f,h);
    end
end