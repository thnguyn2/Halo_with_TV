function x=nlcg(y,params,x0)
%Non-linear conjugatte gradient with line-search
%x=argmin ||y-M*W*x||_2^2+lambda*|V*x|_p + alpha*TV(x) =  argmin f(x)
%However, gradient of |V*u|_p norm can not be computed easily.
%Thus, we replace V*x by u => x=V'*u. Then, we have equivalent problem
%Find u =  argmin ||y-M*W*V'*u||_2^2+lambda*|u|_p + alpha*TV(V'u)
%After we find u, x=V'*u
%Here lambda contains the trade off between data consistency and sparseness
%constrain.
%Author: Tan H. Nguyen
%Writing date: 10/24/2011
%Email: thnguyn2@illinois.edu
%University of Illinois at Urbana-Champaign
%--------------------------------------------------------------------------
%Arguments:
%Input:
%   x0: initial estimation of  the recovered image
%   M: measurement matrix (measuring operator) (in params)
%   y: measurement vectors (2D) 
%   W: WHT operator (in params)
%   D: gradient operator (in params)
%   V: operator defining the domain in which V*x is sparse
%   lambda: trade-off coefficients (in params)

%Reference:
%[1]. Nocedal, 'Numerical Optimization', 2006
%[2]. An introduction to the Conjugate Gradient Method without the
%Argonizaing Pain

%Remarks:
%Note that the second term of f is not differentiable. Thus, we replace it
%by sqrt(uT*u+muy) where muy is a very small number.
%--------------------------------------------------------------------------
M=params.M;
W=params.W;
D=params.D;
V=params.V;

%Load trade of coefficients
lambda=params.lambda;
alpha=params.alpha;

[nr,nc]=size(x0);
%x0=zeros(size(x0));

%Parameters for Line-Search algorithm
params.LSc=1e-4;%The constant c using in Line Search algorithm
                        %Note that the normalizing coefficients is needed
                        %here!!!
params.LSrho=0.6;%Rho used in Line Search algorithm
params.LSMaxiter=100; %Max number of iteration for line-search
params.step0=1;
params.CgTol=1e-5;%Tolerence on the gradient norm to stop iterating 

%Configure Smoothing parameter for |x|_p
params.LpSmooth=1e-8;
params.pNorm=1;

% figure(7);
% imagesc(y);
% title('Measured coefs');


%% Prepare for 1st iteration
u0=V*x0;
n=length(x0(:));
[fk,dc,lpnorm,tv]=fval(y,u0,params);%New objective
 
disp(['Initial Objectives:' ' Obj: ' num2str(fk,'%3.4f') ', dc:' num2str(dc,'%0.3f')...
                ', Spr:' num2str(lpnorm,'%0.3f') ', TV:' num2str(tv,'%0.3f')]);

            
gf0=gfval(y,u0,params);
%------------------------------------------------------------

%disp('Nonlinear conjugate gradient method for optimization');
pk=-gf0;%Initial searching direction
uk=u0;
gfk=gf0;

k=0;
obj_arr=zeros(0);

    while ((k<params.CgMaxiter)&&(norm(gfk,'fro')>params.CgTol))
        [step,lsiter]=ls(y,uk,pk,params,'interp');
        if (lsiter==0)
            params.step0=params.step0/params.LSrho; %If previous step can be found very quickly, 
                                    %then increase step size
        end
        if (lsiter>=2)
            params.step0=params.step0*params.LSrho; %reduce intial searching step with knowledge given in previous step
        end
          
         %State update
         uk1=uk+step*pk;

         %Calculate new gradient
         gfk1=gfval(y,uk1,params);

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

         [fk1,dc,lpnorm,tv]=fval(y,uk1,params);%New objective
        
         %Update all necessary info for next iteration
         pk=pk1;
         gfk=gfk1;
         uk=uk1;
         fk=fk1;
            
         
         obj_arr(end+1)=fk1;
         figure(3);
         plot(obj_arr);
         title('Objective function');
                 
         k=k+1;
         cost=-pk(:)'*gfk(:)/sqrt(pk(:)'*pk(:))/sqrt(gfk(:)'*gfk(:));
         disp(['#' num2str(k) ', step: ' num2str(step) ...
                ', Obj: ' num2str(fk1,'%0.3f') ', dc:' num2str(dc,'%0.2f')...
                ', Spr:' num2str(lpnorm,'%0.2f') ', TV:' num2str(tv,'%0.2f')...
                ', Grad avg: ' num2str(sqrt(gfk(:)'*gfk(:)/nr/nc),'%0.3f')...
                ', Grd Agl:' num2str(grdangle)]);

                 
         figure(2);
         imshow(V'*uk);
         colormap gray;
         colorbar
         
         %Restart if near orthogonal property of the residual is not guarantee
         if ((mod(k,n)==0)||grdangle>0.3)
                %disp('restart..') 
                pk=-gfk;
         end 
    end
    x=V'*uk;
end



