clear all;
close all;
clc;

x =[5.7441;
     1.3800;
     2.2049;
     0.5015;
     0.3528];
[n_x,r]=size(x);
lambda=3-n_x;
P=zeros(n_x,n_x);
P=[0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020;
   -0.0013,    0.0077,    0.0011,    0.0071,    0.0060;
   0.0030,    0.0011,    0.0054,    0.0007,    0.0008;
   -0.0022,    0.0071,    0.0007,    0.0098,    0.0100;
   -0.0020,    0.0060,    0.0008,    0.0100,    0.0123];
A=(chol(P))';  %Cholesky decomposition transform
x_sig=zeros(n_x,2*n_x+1);
x_sig(:,1)=x;
for i=2:n_x+1
    x_sig(:,i)=x+sqrt(lambda+n_x)*A(:,i-1);
    x_sig(:,n_x+i)=x-sqrt(lambda+n_x)*A(:,i-1);
end


n_aug=7;
std_a=0.2;  %process noise standard deviation longitudinal acceleration
std_yawdd=0.2;  %process noise standard deviation yaw acceleration
lambda=3-n_aug;
x_aug=zeros(n_aug,1);
x_aug(1:n_x)=x;
x_aug(n_x+1:end)=0;
P_aug=zeros(n_aug,n_aug);
P_aug(1:n_x,1:n_x)=P;
P_aug(n_x+1:n_aug,n_x+1:n_aug)=[std_a^2,0;
                            0,std_yawdd^2];
A_aug=(chol(P_aug))';
x_aug_sig=zeros(n_aug,2*n_aug+1);
x_aug_sig(:,1)=x_aug;
for i=2:n_aug+1
    x_aug_sig(:,i)=x_aug+sqrt(lambda+n_aug)*A_aug(:,i-1);
    x_aug_sig(:,n_aug+i)=x_aug-sqrt(lambda+n_aug)*A_aug(:,i-1);
end

delta_t=0.1;
%% augmentation 
Xsig_pre=zeros(5,15);
for i=1:(2*n_aug+1)
    p_x=x_aug_sig(1,i);
    p_y=x_aug_sig(2,i);
    v=x_aug_sig(3,i);
    yaw=x_aug_sig(4,i);
    yawd=x_aug_sig(5,i);
    nu_a=x_aug_sig(6,i);
    nu_yawdd=x_aug_sig(7,i);

    if abs(yawd)>0.001
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    else
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    end
    v_p = v;
    yaw_p = yaw + yawd*delta_t;
    yawd_p = yawd;

    %add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    
    Xsig_pre(1,i)=px_p;
    Xsig_pre(2,i)=py_p;
    Xsig_pre(3,i)=v_p;
    Xsig_pre(4,i)=yaw_p;
    Xsig_pre(5,i)=yawd_p;

end

%set weights
w=zeros(2*n_aug+1,1);
w(1)=lambda/(lambda+n_aug);
w(2:end)=0.5/(lambda+n_aug);
%% predicted mean
x_pred=zeros(5,1);
P_pred=zeros(5,5);
for j=1:n_x
    for i=1:length(w)
        xP_matrix(:,i)=w(i)*Xsig_pre(:,i);
    end
    x_pred(j)=sum(xP_matrix(j,:));
end
% predicted covariance
for i=1:length(w)
	x_diff(:,i)=Xsig_pre(:,i)-x_pred;
	pP_matrix(:,:,i)=w(i)* x_diff(:,i)* x_diff(:,i)';
    P_pred=P_pred+ pP_matrix(:,:,i);
end

%% Predict Radar Measurements Assignment
% Measurement Vector
z_pred=zeros(3,1);
r=zeros(1,2*n_aug+1);
phi=zeros(1,2*n_aug+1);
rd=zeros(1,2*n_aug+1);
% measurement noise
std_r=0.3;
std_phi=0.0175;
std_roud=0.1;
R=[std_r^2,0,0;
    0,std_phi^2,0;
    0,0,std_roud^2];
S_pred=zeros(3,3);

for i=1:(2*n_aug+1)
    % State Vector
    px_p=Xsig_pre(1,i);
    py_p=Xsig_pre(2,i);
    v_p=Xsig_pre(3,i);
    yaw_p=Xsig_pre(4,i);

    % Measurement Model
    r(i)=sqrt(px_p^2+py_p^2);
    phi(i)=atan(py_p/px_p);
    rd(i)=(px_p*cos(yaw_p)*v_p+py_p*sin(yaw_p)*v_p)/r(i);
    Z(:,i)=[r(i);
            phi(i);
            rd(i)];
        
    % predicted measurement mean
    z_pred=z_pred+w(i)*Z(:,i);
end
% predicted covariance mean
for i=1:(2*n_aug+1)
    z_diff(:,i)=Z(:,i)-z_pred;
    sP_matrix(:,:,i)=w(i)* z_diff(:,i)* z_diff(:,i)';
    S_pred=S_pred+ sP_matrix(:,:,i);
end
S_pred=S_pred+R;

%% UKF update assignment
% Cross-correlation Matrix
T_pred=zeros(length(x),length(z_pred));
for i=1:(2*n_aug+1)
    x_diff(:,i)=Xsig_pre(:,i)-x_pred;
    z_diff(:,i)=Z(:,i)-z_pred;
    T_matrix(:,:,i)=w(i)*x_diff(:,i)*(z_diff(:,i))';
    T_pred=T_pred+T_matrix(:,:,i);
end
% Kalman gain K
K_pred=T_pred/S_pred;
% Update State
zC=[5.9214;
    0.2187;
    2.0062];
xC=x_pred+K_pred*(zC-z_pred);
% covariance matrix update
PC=P_pred-K_pred*S_pred*K_pred';