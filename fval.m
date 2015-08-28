function obj  = fval(a_gamma,a_tk,params)
   %Compute the objective function E=||arg(gamma)-(delta-h)*arg(a_tk)||^2+lambda*TV(a_tk)+
    lambda = params.lambda;
    D = params.D;
    H = params.H;
    %Measurement error
    evect1 = a_gamma-H*a_tk;
    obj1 = sum(evect1(:).*conj(evect1(:)));
    grad_res = D*a_tk;
    %Total Variation part
    tv = (abs(grad_res(:,:,1)).^2+abs(grad_res(:,:,2)).^2+1e-15).^0.5;
    obj2 = lambda*sum(tv(:));
    obj = (obj1+obj2)/length(a_tk(:));