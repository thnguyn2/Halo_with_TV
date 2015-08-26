function obj=mtimes(d,x) 
    %obj=mtimes(d,x), compute the 2D gradients of x (D*x), produce a 2N x N
    %image of the form [Dx*x;Dy*x] where x is the input image
    x_c1=zeros(size(x));%Horizontal shift
    x_c2=zeros(size(x));
    x_c1(:,2:end)=x(:,1:end-1);
    %x_c2(:,1:end-1)=x(:,2:end);
    x_c2=x;
    %Note that due to symmetry, grad at a pixel in x direction is calculated
    %as the difference between between its right pixel and its left pixel
    %This is very important in order to avoid shifting when calculate the
    %difference between the pixel itself with one of its neighborhood
    
    x_r1=zeros(size(x));%Vertical shift
    x_r2=zeros(size(x));
    x_r1(2:end,:)=x(1:end-1,:);
    %x_r2(1:end-1,:)=x(2:end,:);
    x_r2=x;
    obj=[x_c2-x_c1;x_r2-x_r1];
end