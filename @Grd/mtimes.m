function res=mtimes(d,x) 
   %Compute the results of the Gradient operator for the x and the y
   %dimension.
   %when d.adjoint =0, compute [Dx*x.Dy*x], output result has a dimension
   %of [2 * N * N]
   %when d.adjoint =1, compute the adjoint operator [DxT*x, DyT*x]
   if (d.adjoint)
        %[TBA]
   else
        x_c1=zeros(size(x));%Horizontal shift
        x_c1(:,2:end)=x(:,1:end-1);
        %Note that due to symmetry, grad at a pixel in x direction is calculated
        %as the difference between between its right pixel and its left pixel
        %This is very important in order to avoid shifting when calculate the
        %difference between the pixel itself with one of its neighborhood
        x_r1=zeros(size(x));%Vertical shift
        x_r1(2:end,:)=x(1:end-1,:);
        res=cat(3,x_c1-x,x_r1-x);
   end
   
end