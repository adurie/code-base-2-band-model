#include <iostream>
#include <utility>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include "cunningham_points_adaptive.h"

using namespace Eigen;
using namespace std;

typedef complex<double> dcomp;

Matrix2cd greens(Matrix2cd &OM, Matrix2cd &T)
{
	Matrix2cd D = OM*T.inverse();
	Matrix2cd B = T.inverse();
	Matrix2cd C = T.adjoint();
	Matrix4cd X,O;
	X <<	0,	0,	B(0,0),		B(0,1),
		0,	0,	B(1,0),		B(1,1),
		-C(0,0),-C(0,1),D(0,0),		D(0,1),
		-C(1,0),-C(1,1),D(1,0),		D(1,1);
	ComplexEigenSolver<Matrix4cd> ces;
	ces.compute(X);
	O = ces.eigenvectors();
	Matrix2cd b = O.topRightCorner(2,2);
	Matrix2cd d = O.bottomRightCorner(2,2);
	Matrix2cd GR = b*d.inverse();
	return GR;
}

VectorXcd Rspace(double x, double z, const double a, dcomp E, const int N) {
//...F(0)|NM(n)|F(theta)...
	const double t_11 = 0.5;
	const double t_22 = t_11;
	const double t_12 = 1.;
	const double v1 = -1.6295;
	const double v2 = -2.8;
	const double v3 = -1.6295;
	const double nab = -0.6795;
	const double s = 1.;
	double F = cos(x*a)+cos(z*a);
	Matrix2cd Tsplit;
	Tsplit << (t_11+t_22)*F+sqrt(((t_11-t_22)*(t_11-t_22)+4.*t_12*conj(t_12))*F*F+s*s+2.*s*(t_12+conj(t_12))*F), 0,
			0,	(t_11+t_22)*F-sqrt(((t_11-t_22)*(t_11-t_22)+4.*t_12*conj(t_12))*F*F+s*s+2.*s*(t_12+conj(t_12))*F);

	Matrix2cd T((Matrix2cd() << t_11,t_12,conj(t_12),t_22).finished());
	static const Matrix2cd V2((Matrix2cd() << v2,s,s,v2).finished());
	static const Matrix2cd V3_u((Matrix2cd() << v3+nab,s,s,v3+nab).finished());
	static const Matrix2cd V3_u_AF((Matrix2cd() << v3+nab,-s,-s,v3+nab).finished());
	static const Matrix2cd V3_d((Matrix2cd() << v3-nab,s,s,v3-nab).finished());
	static const Matrix2cd V3_d_AF((Matrix2cd() << v3-nab,-s,-s,v3-nab).finished());
	static const Matrix2cd I((Matrix2cd() << 1.,0.,0.,1.).finished());
	static const Matrix2cd V1_u((Matrix2cd() << v1+nab,s,s,v1+nab).finished());
	static const Matrix2cd V1_d((Matrix2cd() << v1-nab,s,s,v1-nab).finished());
//initialise surface Green's function using mobius transformation for RHS majority greens function
	Matrix2cd OMV3u = E*I-V3_u-Tsplit;
	Matrix2cd GR_u = greens(OMV3u,T);

	Matrix2cd OMV3uAF = E*I-V3_u_AF-Tsplit;
	Matrix2cd GR_u_AF = greens(OMV3uAF, T);

//initialise surface Green's function using mobius transformation for RHS minority greens function
	Matrix2cd OMV3d=E*I-V3_d-Tsplit;
	Matrix2cd GR_d = greens(OMV3d,T);

	Matrix2cd OMV3dAF = E*I-V3_d_AF-Tsplit;
	Matrix2cd GR_d_AF = greens(OMV3dAF, T);

//initialise surface Green's function using mobius transformation for LHS majority greens function
	Matrix2cd OMV1u=E*I-V1_u-Tsplit;
	Matrix2cd GL_u = greens(OMV1u,T);

//initialise surface Green's function using mobius transformation for LHS minority greens function
	Matrix2cd OMV1d=E*I-V1_d-Tsplit;
	Matrix2cd GL_d = greens(OMV1d,T);

	Matrix2cd Rsigma_u_FM, Rsigma_d_FM, Rsigma_u_AF, Rsigma_d_AF;

	Matrix2cd OM = E*I - Tsplit;
	dcomp Fsigma;
	VectorXcd result(N);
	result.fill(0.);
//adlayer layer 2 from layer 1 to spacer thickness, N
	for (int it=0; it != N; ++it){

		GL_u = (OM - V2 -T.adjoint()*GL_u*T).inverse();
		GL_d = (OM - V2 -T.adjoint()*GL_d*T).inverse();
		Rsigma_u_FM = (I-GR_u*T.adjoint()*GL_u*T);
		Rsigma_d_FM = (I-GR_d*T.adjoint()*GL_d*T);
		/* Rsigma_u_AF = (I-GR_d_AF*T.adjoint()*GL_u*T); */
		Rsigma_u_AF = (I-GR_d*T.adjoint()*GL_u*T);
		/* Rsigma_d_AF = (I-GR_u_AF*T.adjoint()*GL_d*T); */
		Rsigma_d_AF = (I-GR_u*T.adjoint()*GL_d*T);
		Fsigma = (1./M_PI)*log((Rsigma_u_FM*Rsigma_d_FM*Rsigma_u_AF.inverse()*Rsigma_d_AF.inverse()).determinant());
		result[it] = Fsigma;
	}
	
	return  result;
}

VectorXd f(const double a, const int N) {
	dcomp i;
	i = -1.;
	i = sqrt(i);
	dcomp E = 0.;
	const double Ef = 0.0;
	const double kT = 8.617342857e-5*300/13.6058;
	VectorXcd result_complex(N);
	result_complex.fill(0.);
	for (int j=0; j!=10; j++){
		E = Ef + (2.*j + 1.)*kT*M_PI*i;
		result_complex = result_complex + kspace(&Rspace, 0, 5e-2, 40, a, E, N);
	}
	VectorXd result_return = result_complex.real();

	return kT*result_return;
}

int main() 
{

	cout<<"Name the data file\n";
	string Mydata;
	getline(cin, Mydata);
	ofstream Myfile;	
	Mydata += ".txt";
	Myfile.open( Mydata.c_str(),ios::trunc );
	const double a = 1.;
	const int N = 50;

	/* dcomp i; */
	/* i = -1; */
	/* i = sqrt(i); */
	/* dcomp Ef = 0. + 1e-5*i; */	

	VectorXd result(N);
	//next line is gamma point only. Bypasses integration
	/* result = f(0,0,params); */
	result = f(a, N);
	result /= 4.*M_PI*M_PI;
	Myfile<<"N , Gamma"<<endl;

	for (int i=0; i < N ; ++i)
		Myfile << i+1 <<" ,  "<< -2.*M_PI*result[i] << endl;

	cout<<"finished!"<<endl;


return 0;
}
