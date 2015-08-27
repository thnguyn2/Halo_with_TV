function obj=Grd(N) %Class constructor for the Gradient Operator
    %Inputs:
    %   N is the size of the image
    %Outputs:
    %   obj: out gradient operator
    power=log(N)/log(2);
    if (power==floor(power)) %Check to see of N is a power of 2
        obj.N=N;
        obj.adjoint=0;%if adjoint=1 then compute DxT*x and DyT*x, else, com[ute Dx*x and Dy*x
        obj=class(obj,'Grd');%Create an object of class Hadamard with struct obj;
    else
        error('N is not a power of 2. Not supported...');
    end
end