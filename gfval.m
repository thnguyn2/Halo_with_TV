function [grad_res,mismatch_grad,tv_grad] = gfval(a_gamma,a_tk,params)
%Compute df/dt at tk in order to solve for tk given the current estimation
%of gk and the current solution tk
%The objective function is f = ||arg(gamma)-Hi.arg(tk)||^2 + lambda*sum_over_r{[(Dx(arg(tk))^2 + Dy(arg(tk))^2 +1e-15)]^0.5}
%   a_gamma = arg(gamma)
%   a_tk = arg(tk)
%Author: Tan H. Nguyen
    lambda = params.lambda;
    D = params.D;
    H = params.H; %Filtering operator
    %Compute the derivative of the 1st term
    mismatch_grad = 2*(H'*(H*a_tk-a_gamma)); %Gradient of the ||a_gamma-H.a_tk||^2 term
    G_val = D*a_tk;                           %Gradient results
    Gx = G_val(:,:,1);
    Gy = G_val(:,:,2);
    tv_grad = -lambda*(Gx+Gy)./(Gx.^2+Gy.^2+1e-15).^(0.5);
    grad_res = mismatch_grad+tv_grad;
    
    
    