function res = mtimes(H,x)
%Compute H*x when adjoint = 0 or H'*x when adjoint = 1
hf = H.hf;
xf = H.F*x;
if (H.adjoint) 
    %Compute H'*x
    res = H.F'*(conj(hf).*xf);
else
    %Compute the filtering result Hx = h*x
    res = H.F'*(hf.*xf); %Fourier transform of the fil
end