function  res = TV()
%res = TV()
%
% Implements a total variation prior TV(I) = summing over pixels
% sqrt(|grad_x(I)|^2+eps) + sqrt(|grad_y(I)|^2+eps) 
%
% (c) Michael Lustig 2007

res.adjoint = 0;
res = class(res,'TV');

