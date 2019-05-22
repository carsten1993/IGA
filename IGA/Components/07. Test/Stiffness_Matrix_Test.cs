using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace IGA
{
    public class Stiffness_Matrix_Test : GH_Component
    {

        public Stiffness_Matrix_Test()
          : base("Stiffness Matrix Test", "stiffness_matrix_test",
              "Calculate the stiffness matrix for the patch",
              "IGA", "07. Test")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddPointParameter("Control points", "B", "Control points as a list", GH_ParamAccess.list);
            pManager.AddNumberParameter("Knot vector csi", "knotsCsi", "Knot vector csi", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Degree csi", "p", "Degree csi", GH_ParamAccess.item, 0);
            pManager.AddNumberParameter("Knot vector eta", "knotsEta", "Knot vector eta", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Degree eta", "q", "Degree eta", GH_ParamAccess.item, 0);
            pManager.AddNumberParameter("Knot vector zeta", "knotsZeta", "Knot vector zeta", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Degree zeta", "r", "Degree zeta", GH_ParamAccess.item, 0);
            pManager.AddGenericParameter("Material", "material", "The material to be disassebbled", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Full stiffness matrix", "K_full", "Full global stiffness matrix", GH_ParamAccess.item);
            pManager.AddGenericParameter("Full force vector", "P_full", "Full global force vector", GH_ParamAccess.item);
            pManager.AddGenericParameter("Stiffness matrix", "K", "Global stiffness matrix", GH_ParamAccess.item);
            pManager.AddGenericParameter("Force vector", "P", "Global force vector", GH_ParamAccess.item);
            pManager.AddNumberParameter("Displacement list", "d_list", "Displacement list", GH_ParamAccess.list);
            pManager.AddPointParameter("New control points", "newB", "New control points as a list", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<Point3d> controlPoints = new List<Point3d>();
            List<double> knotsCsi = new List<double>();
            int p = 0;
            List<double> knotsEta = new List<double>();
            int q = 0;
            List<double> knotsZeta = new List<double>();
            int r = 0;
            Material material = new Material();            

            if (!DA.GetDataList(0, controlPoints)) return;
            if (!DA.GetDataList(1, knotsCsi)) return;
            if (!DA.GetData(2, ref p)) return;
            if (!DA.GetDataList(3, knotsEta)) return;
            if (!DA.GetData(4, ref q)) return;
            if (!DA.GetDataList(5, knotsZeta)) return;
            if (!DA.GetData(6, ref r)) return;
            if (!DA.GetData(7, ref material)) return;

            int n = knotsCsi.Count - p - 1;
            int m = knotsEta.Count - q - 1;
            int l = knotsZeta.Count - r - 1;
            int nel = (n - p) * (m - q) * (l - r);
            int nnp = n * m * l;
            int nen = (p + 1) * (q + 1) * (r + 1);
            double E_Y = material.GetEmodulus();
            double v_P = material.GetPoission();
            double df = material.GetDensityForce();

            if (p > n)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Knot vector in csi-direction does not match degree p");
                return;
            }
            if (q > m)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Knot vector in eta-direction does not match degree q");
                return;
            }
            if (r > l)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Knot vector in zeta-direction does not match degree r");
                return;
            }
            if (nnp != controlPoints.Count)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Number of control points does not match the number of basis function for the given degrees");
                return;
            }

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<List<List<Point3d>>> B = CreateControlPointList(controlPoints, n, m, l);
            (Matrix<double> K_full, Vector<double> P_full) = CreateStiffnessMatrixLoadVector(p, q, r, knotsCsi, knotsEta, knotsZeta, B, E_Y, v_P, df);
            List<int> res1 = new List<int>(){ 1, 1, 1, 1};
            List<int> res2 = new List<int>() {7, 1, 1, 1 };
            List<int> res3 = new List<int>() {13, 1, 1, 1 };
            List<int> res4 = new List<int>() { 19, 1, 1, 1 };
            List<int> res5 = new List<int>() { 25, 1, 1, 1 };
            List<int> res6 = new List<int>() { 31, 1, 1, 1 };
            List<int> res7 = new List<int>() { 37, 1, 1, 1 };
            List<int> res8 = new List<int>() { 43, 1, 1, 1 };
            List<int> res9 = new List<int>() { 49, 1, 1, 1 };
            List<List<int>> res = new List<List<int>>() { res1, res2, res3, res4, res5, res6, res7, res8, res9 };
            (Matrix<double> K, Vector<double> P) = ReshapeStiffnessMatrixLoadVector(K_full, P_full, res, nnp);
            Vector<double> d = K.Inverse() * P;
            List<double> d_list = VectorToList_double(d);
            List<Point3d> newB = CreateNewControlPointList(p, q, r, n, m, l, d, B);


            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, K_full);
            DA.SetData(1, P_full);
            DA.SetData(2, K);
            DA.SetData(3, P);
            DA.SetDataList(4, d_list);
            DA.SetDataList(5, newB);

        }


        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        Vector<double> ListToVector_double(List<double> list)
        {
            Vector<double> vector = DenseVector.OfArray(new double[list.Count]);

            int i = 0;
            foreach (double d in list)
            {
                vector[i] = d;
                i++;
            }

            return vector;
        }

        List<double> VectorToList_double(Vector<double> vec)
        {
            List<double> list = new List<double>();

            foreach (double d in vec)
            {
                list.Add(d);
            }

            return list;
        }

        Matrix<double> ListToMatrix_double(List<List<double>> list)
        {
            int row = list.Count;
            int col = list[0].Count;
            Matrix<double> matrix = DenseMatrix.OfArray(new double[row, col]);

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    matrix[i, j] = list[i][j];
                }
            }

            return matrix;
        }

        (List<double>, List<double>) Gauss(int ngp)
        {

            List<double> list1 = new List<double>(); //Sample Coord in order(exterior-middle-center)
            List<double> list2 = new List<double>(); //Weights


            switch (ngp)
            {
                case 1:
                    list1.Add(0.0);
                    list2.Add(2.0);
                    return (list1, list2);

                case 2:
                    double C = 0.5773502691;
                    list1.Add(-C);
                    list1.Add(C);
                    list2.Add(1.0);
                    list2.Add(1.0);
                    return (list1, list2);

                case 3:
                    double C1 = 0.774596669241483;
                    double C2 = 0.0;
                    double W1 = 0.555555555555555;
                    double W2 = 0.88888888888888;
                    list1.Add(-C1);
                    list1.Add(C2);
                    list1.Add(C1);
                    list2.Add(W1);
                    list2.Add(W2);
                    list2.Add(W1);
                    return (list1, list2);
                case 4:
                    double C11 = 0.861136311594053;
                    double C22 = 0.339981043584856;
                    double W11 = 0.347854845137454;
                    double W22 = 0.652145154862546;
                    list1.Add(-C11);
                    list1.Add(-C22);
                    list1.Add(C22);
                    list1.Add(C11);
                    list2.Add(W11);
                    list2.Add(W22);
                    list2.Add(W22);
                    list2.Add(W11);
                    return (list1, list2);

            }
            return (list1, list2);
        }

        List<List<List<Point3d>>> CreateControlPointList(List<Point3d> controlPoints, int n, int m, int l)
        {
            List<List<List<Point3d>>> B = new List<List<List<Point3d>>>();

            for (int k = 0; k < l; k++)
            {
                List<List<Point3d>> B_i_j = new List<List<Point3d>>();
                for (int j = 0; j < m; j++)
                {
                    List<Point3d> B_i = new List<Point3d>();
                    for (int i = 0; i < n; i++)
                    {
                        B_i.Add(controlPoints[k * n * m + j * n + i]);
                    }
                    B_i_j.Add(B_i);
                }
                B.Add(B_i_j);
            }

            return B;
        }

        List<Point3d> CreateNewControlPointList(int p, int q, int r, int n, int m, int l, Vector<double> d, List<List<List<Point3d>>> B)
        {
            List<Point3d> newB = new List<Point3d>();

            int id;
            for (int k = 0; k < l; k++)
            {
                for (int j = 0; j < m; j++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        id = k * n * m + j * n + i;
                        newB.Add(B[k][j][i] + new Point3d(d[id * 3], d[id * 3 + 1], d[id * 3 + 2]));
                    }
                }
            }

            return newB;
        }

        double BasisFunction(List<double> knotsCsi, int i, int p, double csi)
        {
            double sum = 0;

            if (p == 0)
            {
                if (csi >= knotsCsi[i] && csi <= knotsCsi[i + 1])
                {
                    return 1;
                }
                else return 0;
            }
            else
            {
                double d1 = knotsCsi[i + p] - knotsCsi[i];
                double d2 = knotsCsi[i + p + 1] - knotsCsi[i + 1];

                if (d1 != 0 && d2 == 0)
                {
                    double e1 = csi - knotsCsi[i];
                    sum = e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1;
                }
                else if (d1 == 0 && d2 != 0)
                {
                    double e2 = knotsCsi[i + p + 1] - csi;
                    sum = e2 * BasisFunction(knotsCsi, i + 1, p - 1, csi) / d2;
                }
                else if (d1 != 0 && d2 != 0)
                {
                    double e1 = csi - knotsCsi[i];
                    double e2 = knotsCsi[i + p + 1] - csi;

                    if (csi == knotsCsi[i + 1])
                    {
                        sum = (e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1);
                    }
                    else
                    {
                        sum = (e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1) + (e2 * BasisFunction(knotsCsi, i + 1, p - 1, csi) / d2);
                    }
                }
                else return 0;

                return sum;
            }
        }

        double DerivativeBasisFunction(List<double> knotsCsi, int i, int p, double csi)
        {
            double sum = 0;

            if (p == 0)
            {
                if (csi >= knotsCsi[i] && csi <= knotsCsi[i + 1])
                {
                    return 1;
                }
                else return 0;
            }
            else
            {
                double d1 = knotsCsi[i + p] - knotsCsi[i];
                double d2 = knotsCsi[i + p + 1] - knotsCsi[i + 1];

                if (d1 != 0 && d2 == 0)
                {
                    double e1 = p;
                    sum = e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1;
                }
                else if (d1 == 0 && d2 != 0)
                {
                    double e2 = p;
                    sum = -e2 * BasisFunction(knotsCsi, i + 1, p - 1, csi) / d2;
                }
                else if (d1 != 0 && d2 != 0)
                {
                    double e1 = p;
                    double e2 = p;

                    if (csi == knotsCsi[i + 1])
                    {
                        sum = (e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1);
                    }
                    else
                    {
                        sum = (e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1) - (e2 * BasisFunction(knotsCsi, i + 1, p - 1, csi) / d2);
                    }
                }
                else return 0;

                return sum;
            }
        }

        (List<List<int>>, List<List<int>>) CreateINN_IEN(int p, int q, int r, int n, int m, int l)
        {
            List<List<int>> INN = new List<List<int>>();
            List<List<int>> IEN = new List<List<int>>();

            int A = 0, B;
            for (int k = 1; k <= l; k++)
            {
                for (int j = 1; j <= m; j++)
                {
                    for (int i = 1; i <= n; i++)
                    {
                        List<int> tempINN = new List<int>();
                        tempINN.Add(i);
                        tempINN.Add(j);
                        tempINN.Add(k);
                        INN.Add(tempINN);
                        A++;

                        if (i >= p + 1 && j >= q + 1 && k >= r + 1)
                        {
                            List<int> tempIEN = new List<int>();
                            for (int kloc = 0; kloc <= r; kloc++)
                            {
                                for (int jloc = 0; jloc <= q; jloc++)
                                {
                                    for (int iloc = 0; iloc <= p; iloc++)
                                    {
                                        B = A - kloc * n * m - jloc * n - iloc;
                                        tempIEN.Add(B);
                                    }
                                }
                            }
                            IEN.Add(tempIEN);
                        }
                    }
                }
            }

            return (INN, IEN);
        }

        List<int> GetNURBSCoord(List<List<int>> INN, List<List<int>> IEN, int e)
        {
            int ni = INN[IEN[e - 1][0] - 1][0];
            int nj = INN[IEN[e - 1][0] - 1][1];
            int nk = INN[IEN[e - 1][0] - 1][2];

            return new List<int>(){ ni, nj, nk };
        }

        List<double> GetParametricCoord(List<int> nCoord, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta, List<double> csiTilde)
        {
            int ni = nCoord[0];
            int nj = nCoord[1];
            int nk = nCoord[2];

            double csi = ((knotsCsi[ni] - knotsCsi[ni - 1]) * csiTilde[0] + (knotsCsi[ni] + knotsCsi[ni - 1])) / 2;
            double eta = ((knotsEta[nj] - knotsEta[nj - 1]) * csiTilde[1] + (knotsEta[nj] + knotsEta[nj - 1])) / 2;
            double zeta = ((knotsZeta[nk] - knotsZeta[nk - 1]) * csiTilde[2] + (knotsZeta[nk] + knotsZeta[nk - 1])) / 2;

            return new List<double>() { csi, eta, zeta };
        }

        (List<int>, List<double>) GetCoordinates(List<List<int>> INN, List<List<int>> IEN, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta, List<double> csiTilde, int e)
        {
            List<int> nCoord = GetNURBSCoord(INN, IEN, e);
            List<double> pCoord = GetParametricCoord(nCoord, knotsCsi, knotsEta, knotsZeta, csiTilde);

            return (nCoord, pCoord);
        }

        (List<List<double>>, List<List<double>>, List<List<double>>) UnivariateBasisFuntion(int p, int q, int r, List<int> nCoord, List<double> pCoord, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta)
        {
            List<List<double>> NdN = new List<List<double>>();
            List<double> N = new List<double>();
            List<double> dN = new List<double>();
            List<List<double>> MdM = new List<List<double>>();
            List<double> M = new List<double>();
            List<double> dM = new List<double>();
            List<List<double>> LdL = new List<List<double>>();
            List<double> L = new List<double>();
            List<double> dL = new List<double>();

            int ni = nCoord[0];
            double csi = pCoord[0];
            int nj = nCoord[1];
            double eta = pCoord[1];
            int nk = nCoord[2];
            double zeta = pCoord[2];

            for (int i = ni - p; i <= ni; i++)
            {
                N.Add(BasisFunction(knotsCsi, i - 1, p, csi));
                dN.Add(DerivativeBasisFunction(knotsCsi, i - 1, p, csi));
            }
            NdN.Add(N);
            NdN.Add(dN);

            for (int j = nj - q; j <= nj; j++)
            {
                M.Add(BasisFunction(knotsEta, j - 1, q, eta));
                dM.Add(DerivativeBasisFunction(knotsEta, j - 1, q, eta));
            }
            MdM.Add(M);
            MdM.Add(dM);

            for (int k = nk - r; k <= nk; k++)
            {
                L.Add(BasisFunction(knotsZeta, k - 1, r, zeta));
                dL.Add(DerivativeBasisFunction(knotsZeta, k - 1, r, zeta));
            }
            LdL.Add(L);
            LdL.Add(dL);

            return (NdN, MdM, LdL);
        }

        (List<double>, List<List<double>>) TrivariateBasisFunction(int p, int q, int r, List<List<double>> NdN, List<List<double>> MdM, List<List<double>> LdL)
        {
            List<double> R = new List<double>();
            List<List<double>> dR = new List<List<double>>();
            List<double> dR_dCsi = new List<double>();
            List<double> dR_dEta = new List<double>();
            List<double> dR_dZeta = new List<double>();

            for (int k = 0; k <= r; k++)
            {
                for (int j = 0; j <= q; j++)
                {
                    for (int i = 0; i <= p; i++)
                    {
                        R.Add(NdN[0][i] * MdM[0][j] * LdL[0][k]);
                        dR_dCsi.Add(NdN[1][i] * MdM[0][j] * LdL[0][k]);
                        dR_dEta.Add(NdN[0][i] * MdM[1][j] * LdL[0][k]);
                        dR_dZeta.Add(NdN[0][i] * MdM[0][j] * LdL[1][k]);
                    }
                }
            }

            dR.Add(dR_dCsi);
            dR.Add(dR_dEta);
            dR.Add(dR_dZeta);

            return (R, dR);
        }

        (Matrix<double>, Matrix<double>) CreateGradients(int p, int q, int r, List<int> nCoord, List<List<List<Point3d>>> B, List<List<double>> dR_dCsi, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta)
        {
            Matrix<double> dx_dCsi = DenseMatrix.OfArray(new double[3, 3]);
            Matrix<double> dCsi_dCsiTilde = DenseMatrix.OfArray(new double[3, 3]);

            int ni = nCoord[0];
            int nj = nCoord[1];
            int nk = nCoord[2];

            dCsi_dCsiTilde[0, 0] = (knotsCsi[ni] - knotsCsi[ni - 1]) / 2;
            dCsi_dCsiTilde[1, 1] = (knotsEta[nj] - knotsEta[nj - 1]) / 2;
            dCsi_dCsiTilde[2, 2] = (knotsZeta[nk] - knotsZeta[nk - 1]) / 2;

            Point3d point;
            int loc = 0;
            for (int k = 0; k <= r; k++)
            {
                for (int j = 0; j <= q; j++)
                {
                    for (int i = 0; i <= p; i++)
                    {
                        point = B[nk - r + k - 1][nj - q + j - 1][ni - p + i - 1];
                        List<double> pointList = new List<double>() { point.X, point.Y, point.Z };

                        for (int a = 0; a < 3; a++)
                        {
                            for (int b = 0; b < 3; b++)
                            {
                                dx_dCsi[a, b] += dR_dCsi[b][loc] * pointList[a];
                            }
                        }
                        loc++;
                    }
                }
            }

            return (dx_dCsi, dCsi_dCsiTilde);
        }

        List<List<double>> TrivariateBasisFunctionPhysical(int nen, List<List<double>> dR_dCsi, Matrix<double> dCsi_dx)
        {
            List<List<double>> dR_dx = new List<List<double>>();
            List<double> dR_dx1 = new List<double>();
            List<double> dR_dx2 = new List<double>();
            List<double> dR_dx3 = new List<double>();

            double temp;
            for (int loc = 0; loc < nen; loc++)
            {
                List<double> tempA = new List<double>();
                for (int a = 0; a < 3; a++)
                {
                    temp = 0;
                    for (int b = 0; b < 3; b++)
                    {
                        temp += dR_dCsi[b][loc] * dCsi_dx[b, a];
                    }
                    tempA.Add(temp);
                }
                dR_dx1.Add(tempA[0]);
                dR_dx2.Add(tempA[1]);
                dR_dx3.Add(tempA[2]);
            }

            dR_dx.Add(dR_dx1);
            dR_dx.Add(dR_dx2);
            dR_dx.Add(dR_dx3);

            return dR_dx;
        }

        (List<double>, List<List<double>>, double) GetShapeFunctions(int p, int q, int r, List<int> nCoord, List<double> pCoord, int nen, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta, List<List<List<Point3d>>> B)
        {
            (List<List<double>> NdN, List<List<double>> MdM, List<List<double>> LdL) = UnivariateBasisFuntion(p, q, r, nCoord, pCoord, knotsCsi, knotsEta, knotsZeta);
            (List<double> R, List<List<double>> dR_dCsi) = TrivariateBasisFunction(p, q, r, NdN, MdM, LdL);
            (Matrix<double> dx_dCsi, Matrix<double> dCsi_dCsiTilde) = CreateGradients(p, q, r, nCoord, B, dR_dCsi, knotsCsi, knotsEta, knotsZeta);
            Matrix<double> dCsi_dx = dx_dCsi.Inverse();
            List<List<double>> dR_dx = TrivariateBasisFunctionPhysical(nen, dR_dCsi, dCsi_dx);
            Matrix<double> J_mat = dx_dCsi * dCsi_dCsiTilde;
            double J = J_mat.Determinant();

            return (R, dR_dx, J);
        }

        Matrix<double> CreateCMatrix(double lambda, double mu)
        {
            Matrix<double> C = DenseMatrix.OfArray(new double[6, 6]{
                {lambda + 2*mu, lambda, lambda, 0, 0, 0},
                {lambda, lambda + 2*mu, lambda, 0, 0, 0},
                {lambda, lambda, lambda + 2*mu, 0, 0, 0},
                {0, 0, 0, mu, 0, 0},
                {0, 0, 0, 0, mu, 0},
                {0, 0, 0, 0, 0, mu}});

            return C;
        }

        Matrix<double> CreateBMatrix(List<List<double>> dR_dx, int nen)
        {
            Matrix<double> B = DenseMatrix.OfArray(new double[6, 3 * nen]);

            int counter = 0;
            for (int i = 0; i < 3 * nen; i += 3)
            {
                B[0, i] = dR_dx[0][counter];
                B[1, i + 1] = dR_dx[1][counter];
                B[2, i + 2] = dR_dx[2][counter];
                B[3, i + 1] = dR_dx[2][counter];
                B[3, i + 2] = dR_dx[1][counter];
                B[4, i] = dR_dx[2][counter];
                B[4, i + 2] = dR_dx[0][counter];
                B[5, i] = dR_dx[1][counter];
                B[5, i + 1] = dR_dx[0][counter];
                counter++;
            }

            return B;
        }

        Matrix<double> CreateGaussPointElementStiffnessMatrix(List<List<double>> dR_dx, int nen, double lambda, double mu, double modJ)
        {
            Matrix<double> C = CreateCMatrix(lambda, mu);
            Matrix<double> B = CreateBMatrix(dR_dx, nen);
            Matrix<double> Kgpe = B.Transpose() * C * B * modJ;

            return Kgpe;
        }

        Vector<double> CreateGaussPointElementLoadVector(List<double> R, int nen, double df, double modJ)
        {
            Vector<double> Pgpe = DenseVector.OfArray(new double[3 * nen]);

            for (int i = 0; i < nen; i++)
            {
                Pgpe[i * 3 + 2] = -df * R[i] * modJ;
            }

            return Pgpe;
        }

        (Matrix<double>, Vector<double>) CreateElementStiffnessMatrixLoadVector(int p, int q, int r, int nen, List<int> nCoord, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta, List<List<List<Point3d>>> B, double lambda, double mu, double df)
        {
            Matrix<double> Ke = DenseMatrix.OfArray(new double[3 * nen, 3 * nen]);
            Vector<double> Pe = DenseVector.OfArray(new double[3 * nen]);

            int ngpCsi = p + 1;
            int ngpEta = q + 1;
            int ngpZeta = r + 1;
            (List<double> csiTildeCoord, List<double> weightCsi) = Gauss(ngpCsi);
            (List<double> etaTildeCoord, List<double> weightEta) = Gauss(ngpEta);
            (List<double> zetaTildeCoord, List<double> weightZeta) = Gauss(ngpZeta);

            for (int k = 0; k < ngpZeta; k++)
            {
                for (int j = 0; j < ngpEta; j++)
                {
                    for (int i = 0; i < ngpCsi; i++)
                    {
                        List<double> csiTilde = new List<double>() { csiTildeCoord[i], etaTildeCoord[j], zetaTildeCoord[k] };
                        List<double> pCoord = GetParametricCoord(nCoord, knotsCsi, knotsEta, knotsZeta, csiTilde);
                        (List<double> R, List<List<double>> dR_dx, double J) = GetShapeFunctions(p, q, r, nCoord, pCoord, nen, knotsCsi, knotsEta, knotsZeta, B);
                        double modJ = J * weightCsi[i] * weightEta[j] * weightZeta[k];
                        Ke += CreateGaussPointElementStiffnessMatrix(dR_dx, nen, lambda, mu, modJ);
                        Pe += CreateGaussPointElementLoadVector(R, nen, df, modJ);
                    }
                }
            }

            return (Ke, Pe);
        }
        
        (Matrix<double>, Vector<double>) CreateStiffnessMatrixLoadVector(int p, int q, int r, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta, List<List<List<Point3d>>> B, double E_Y, double v_P, double df)
        {
            int n = knotsCsi.Count - p - 1;
            int m = knotsEta.Count - q - 1;
            int l = knotsZeta.Count - r - 1;
            int nel = (n - p) * (m - q) * (l - r);
            int nnp = n * m * l;
            int nen = (p + 1) * (q + 1) * (r + 1);

            double lambda = v_P * E_Y / ((1 + v_P) * (1 - 2 * v_P));
            double mu = E_Y / (2 * (1 + v_P));

            (List<List<int>> INN, List<List<int>> IEN) = CreateINN_IEN(p, q, r, n, m, l);

            Matrix<double> K = DenseMatrix.OfArray(new double[3 * nnp, 3 * nnp]);
            Matrix<double> Ke = DenseMatrix.OfArray(new double[3 * nen, 3 * nen]);
            Vector<double> P = DenseVector.OfArray(new double[3 * nnp]);
            Vector<double> Pe = DenseVector.OfArray(new double[3 * nen]);

            for (int e = 1; e <= nel; e++)
            {
                List<int> nCoord = GetNURBSCoord(INN, IEN, e);
                int ni = nCoord[0];
                int nj = nCoord[1];
                int nk = nCoord[2];

                if (knotsCsi[ni] == knotsCsi[ni-1] || knotsEta[nj] == knotsEta[nj - 1] || knotsZeta[nk] == knotsZeta[nk - 1])
                {
                    continue;
                }

                (Ke, Pe) = CreateElementStiffnessMatrixLoadVector(p, q, r, nen, nCoord, knotsCsi, knotsEta, knotsZeta, B, lambda, mu, df);

                int localI1, localI2, localI3, localJ1, localJ2, localJ3, globalI1, globalI2, globalI3, globalJ1, globalJ2, globalJ3, globalFI, globalFJ;
                for (int i = 0; i < 3 * nen; i += 3)
                {
                    globalFI = nen - 1 - (i/3);

                    localI1 = i;
                    localI2 = i + 1;
                    localI3 = i + 2;

                    globalI1 = (IEN[e - 1][globalFI] - 1) * 3;
                    globalI2 = (IEN[e - 1][globalFI] - 1) * 3 + 1;
                    globalI3 = (IEN[e - 1][globalFI] - 1) * 3 + 2;

                    P[globalI1] += Pe[localI1];
                    P[globalI2] += Pe[localI2];
                    P[globalI3] += Pe[localI3];

                    for (int j = 0; j < 3 * nen; j += 3)
                    {
                        globalFJ = nen - 1 - (j / 3);
                        
                        localJ1 = j;
                        localJ2 = j + 1;
                        localJ3 = j + 2;
                        
                        globalJ1 = (IEN[e - 1][globalFJ] - 1) * 3;
                        globalJ2 = (IEN[e - 1][globalFJ] - 1) * 3 + 1;
                        globalJ3 = (IEN[e - 1][globalFJ] - 1) * 3 + 2;

                        K[globalI1, globalJ1] += Ke[localI1, localJ1];
                        K[globalI1, globalJ2] += Ke[localI1, localJ2];
                        K[globalI1, globalJ3] += Ke[localI1, localJ3];

                        K[globalI2, globalJ1] += Ke[localI2, localJ1];
                        K[globalI2, globalJ2] += Ke[localI2, localJ2];
                        K[globalI2, globalJ3] += Ke[localI2, localJ3];

                        K[globalI3, globalJ1] += Ke[localI3, localJ1];
                        K[globalI3, globalJ2] += Ke[localI3, localJ2];
                        K[globalI3, globalJ3] += Ke[localI3, localJ3];

                    }
                }

            }

            return (K, P);

        }

        (Matrix<double>, Vector<double>) ReshapeStiffnessMatrixLoadVector(Matrix<double> K_full, Vector<double> P_full, List<List<int>> res, int nnp)
        {
            Matrix<double> K = DenseMatrix.OfMatrix( K_full );
            Vector<double> P = DenseVector.OfVector( P_full );

            int id, resx, resy, resz;
            foreach (List<int> list in res)
            {
                id = 3 * (list[0] - 1);
                resx = list[1];
                resy = list[2];
                resz = list[3];                

                if ( resx == 1)
                {
                    P[id] = 0;
                    for (int i = 0; i < 3 * nnp; i++)
                    {
                        K[id, i] = 0;
                        K[i, id] = 0;
                        K[id, id] = 1;
                    }
                }
                if (resy == 1)
                {
                    P[id + 1] = 0;
                    for (int i = 0; i < 3 * nnp; i++)
                    {
                        K[id + 1, i] = 0;
                        K[i, id + 1] = 0;
                        K[id + 1, id + 1] = 1;
                    }
                }
                if (resz == 1)
                {
                    P[id + 2] = 0;
                    for (int i = 0; i < 3 * nnp; i++)
                    {
                        K[id + 2, i] = 0;
                        K[i, id + 2] = 0;
                        K[id + 2, id + 2] = 1;
                    }
                }                
            }

            return (K, P);
        }

        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                //You can add image files to your project resources and access them like this:
                // return Resources.IconForThisComponent;
                return null;
            }
        }

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("5d47d664-7a90-4fb0-a676-3459650d642b"); }
        }
    }
}