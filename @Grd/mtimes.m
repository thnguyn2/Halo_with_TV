function res=mtimes(d,x) 
   %Compute the results of the Gradient operator for the x and the y
   %dimension.
   %when d.adjoint =0, compute [Dx*x.Dy*x], output result has a dimension
   %of [2 * N * N]
   %when d.adjoint =1, compute the adjoint operator [DxT*x, DyT*x]
   if (d.adjoint)
   
   else
       resy = x([2:end,end],:)-x;
       resx = x(:,[2:end,end])-x;
       res = cat(3,resx,resy);
   end
   
end