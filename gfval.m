function [grad_res,mismatch_grad,tv_grad] = gfval(a_gamma,a_tk,params)
%Compute df/dt at tk in order to solve for tk given the current estimation
%of gk and the current solution tk
%The objective function is f = ||arg(gamma)-Hi.arg(tk)||^2 + lambda*sum_over_r{[(Dx(arg(tk))^2 + Dy(arg(tk))^2 +1e-15)]^0.5}
%   a_gamma = arg(gamma)
%   a_tk = arg(tk)
%Author: Tan H. Nguyen
    lambda = params.lambda;
    TV = params.TV;
    H = params.H; %Filtering operator
    %Compute the derivative of the 1st term
    mismatch_grad = 2*(H'*(H*a_tk-a_gamma)); %Gradient of the ||a_gamma-H.a_tk||^2 term
    Dx =TV*a_tk;
    G = Dx.*(Dx.*conj(Dx) + 1e-15).^(-0.5);%Derivative of the TV
    tv_grad = lambda*(TV'*G);    %See Tan's notebook for the adjoint operator of TV
    grad_res = mismatch_grad+tv_grad;
  
    
    