function res = mtimes(H,x)
%Compute H*x when adjoint = 0 or H'*x when adjoint = 1
if (H.adjoint) 

else
    %Compute the filtering result Hx = h*x
    res = H.F'*((H.hf).*(H.F*x)); %Fourier transform of the fil
end