function obj=Grd(N) %Class constructor for Gradient class for N is the size of the image
    power=log(N)/log(2);
    if (power==floor(power))
        obj.N=N;
        obj.adjoint=0;%if adjoint=1 then compute DxT*x and DyT*x
        obj=class(obj,'Grd');%Create an object of class Hadamard with struct obj;
    end
end