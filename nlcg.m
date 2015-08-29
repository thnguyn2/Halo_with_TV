function x=nlcg(y,params,x0)
%Non-linear conjugatte gradient with line-search
%x=argmin ||y-H*x||_2^2+ alpha*TV(x) =  argmin f(x)
%However, gradient of |V*u|_p norm can not be computed easily.
%Here lambda contains the trade off between data consistency and sparseness
%constrain.
%Author: Tan H. Nguyen
%Writing date: 10/24/2011
%Email: thnguyn2@illinois.edu
%University of Illinois at Urbana-Champaign
%--------------------------------------------------------------------------
%Arguments:
%Input:
%   x0: initial estimation of the recovered image
%   y: measured image
%   params: a struct containing all the constants and operators
%Reference:
%[1]. Nocedal, 'Numerical Optimization', 2006
%[2]. An introduction to the Conjugate Gradient Method without the
%Argonizaing Pain

%Remarks:
%Note that the second term of f is not differentiable. Thus, we replace it
%by sqrt(uT*u+muy) where muy is a very small number.
%--------------------------------------------------------------------------
F=params.F;
D=params.D;
H=params.H;
lambda = params.lambda;

[nr,nc]=size(x0);

%Parameters for Line-Search algorithm
params.LSc=1e-4;%The constant c using in Line Search algorithm
                        %Note that the normalizing coefficients is needed
                        %here!!!
params.LSrho=0.6;%Rho used in Line Search algorithm
params.LSMaxiter=100; %Max number of iteration for line-search
params.step0=1;
params.CgTol=1e-5;%Tolerence on the gradient norm to stop iterating 
params.CgMaxiter = 20;
%Configure Smoothing parameter for |x|_p
params.LpSmooth=1e-8;
params.pNorm=1;


% figure(7);
% imagesc(y);
% title('Measured coefs');


%% Prepare for 1st iteration
[fk,dc,tv]=fval(y,x0,params);%New objective
disp(['Initial Objectives:' ' Obj: ' num2str(fk,'%3.4f') ', dc: ' num2str(dc,'%0.3f'), ', TV: ' num2str(tv,'%0.3f')]);
gf0 =gfval(y,x0,params);
%------------------------------------------------------------

disp('Nonlinear conjugate gradient method for optimization');
pk=-gf0;%Initial searching direction
xk=x0;
gfk=gf0;

k=0;
obj_arr=zeros(0);

    while ((k<params.CgMaxiter)&&(norm(gfk,'fro')>params.CgTol))
        [step,lsiter]=ls(y,xk,pk,params,'back');
        if (lsiter==0)
            params.step0=params.step0/params.LSrho; %If previous step can be found very quickly then increase step size
        end
        if (lsiter>=2)
            params.step0=params.step0*params.LSrho; %reduce intial searching step with knowledge given in previous step
        end
          
         %State update
         xk1=xk+step*pk;

         %Calculate new gradient
         gfk1=gfval(y,xk1,params);

         %Updating coefficients
         beta=(gfk1(:)'*(gfk1(:)-gfk(:)))/(gfk(:)'*gfk(:)+eps); %This is similar to perform 
                                                %restarting once a poor
                                                %direction is met
         beta=max(beta,0);
                                                %New searching direction
         pk1=-gfk1+beta*pk;

         %Compute the condition for restarting
         grdangle=abs(gfk1(:)'*gfk(:))/sqrt(gfk1(:)'*gfk1(:))/...
             sqrt(gfk(:)'*gfk(:));

         [fk1,dc,tv]=fval(y,xk1,params);%New objective
        
         %Update all necessary info for next iteration
         pk=pk1;
         gfk=gfk1;
         xk=xk1;
         fk=fk1;
            
         
         obj_arr(end+1)=fk1;
         figure(3);
         plot(obj_arr);
         title('Objective function');
                 
         k=k+1;
         cost=-pk(:)'*gfk(:)/sqrt(pk(:)'*pk(:))/sqrt(gfk(:)'*gfk(:));
         disp(['#' num2str(k) ', step: ' num2str(step) ...
                ', Obj: ' num2str(fk1,'%0.5f') ', dc:' num2str(dc,'%0.5f')...
                ', TV:' num2str(tv,'%0.5f')...
                ', Grad avg: ' num2str(sqrt(gfk(:)'*gfk(:)/nr/nc),'%0.3f')...
                ', Grd Agl:' num2str(grdangle)]);

                 
         figure(2);
         imshow(xk);
         colormap gray;
         colorbar;drawnow;
         
         %Restart if near orthogonal property of the residual is not guarantee
         if (grdangle>0.3)
                %disp('restart..') 
                pk=-gfk;
         end 
    end
    x=xk;




