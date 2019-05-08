using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace IGA
{
    public class Determinant_Jacobi_Test : GH_Component
    {

        public Determinant_Jacobi_Test()
          : base("Determinant Jacobi Test", "determinant_jacobi_test",
              "Calculates the vector of local shapefunctions R and their derivatives, and the Jacobian determinant J",
              "IGA", "1.")
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
            pManager.AddNumberParameter("Quandrature point", "csiTilde", "Quandrature point [csiTilde, etaTilde, zetaTilde]", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Element", "e", "Element", GH_ParamAccess.item, 1);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddNumberParameter("N", "n", "N", GH_ParamAccess.list);
            pManager.AddNumberParameter("M", "m", "M", GH_ParamAccess.list);
            pManager.AddNumberParameter("L", "l", "L", GH_ParamAccess.list);
            pManager.AddNumberParameter("dN", "dn", "dN", GH_ParamAccess.list);
            pManager.AddNumberParameter("dM", "dm", "dM", GH_ParamAccess.list);
            pManager.AddNumberParameter("dL", "dl", "dL", GH_ParamAccess.list);
            pManager.AddNumberParameter("R", "r", "R", GH_ParamAccess.list);
            pManager.AddNumberParameter("dR_dCsi", "dr_dcsi", "dR_dCsi", GH_ParamAccess.list);
            pManager.AddNumberParameter("dR_dEta", "dr_deta", "dR_dEta", GH_ParamAccess.list);
            pManager.AddNumberParameter("dR_dZeta", "dr_dzeta", "dR_dZeta", GH_ParamAccess.list);
            pManager.AddNumberParameter("dR_dx", "dr_dx", "dR_dx", GH_ParamAccess.list);
            pManager.AddNumberParameter("dR_dy", "dr_dy", "dR_dy", GH_ParamAccess.list);
            pManager.AddNumberParameter("dR_dz", "dr_dz", "dR_dz", GH_ParamAccess.list);
            pManager.AddGenericParameter("dx_dCsi", "dx_dcsi", "dx_dCsi", GH_ParamAccess.item);
            pManager.AddGenericParameter("dCsi_dCsiTilde", "dcsi_dcsitilde", "dCsi_dCsiTilde", GH_ParamAccess.item);
            pManager.AddGenericParameter("Jacobi matrix", "J_mat", "Jacobi matrix", GH_ParamAccess.item);
            pManager.AddNumberParameter("Jacobi determinant", "J", "Jacobi determinant", GH_ParamAccess.item);
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
            List<double> csiTilde = new List<double>();
            int e = 1;

            if (!DA.GetDataList(0, controlPoints)) return;
            if (!DA.GetDataList(1, knotsCsi)) return;
            if (!DA.GetData(2, ref p)) return;
            if (!DA.GetDataList(3, knotsEta)) return;
            if (!DA.GetData(4, ref q)) return;
            if (!DA.GetDataList(5, knotsZeta)) return;
            if (!DA.GetData(6, ref r)) return;
            if (!DA.GetDataList(7, csiTilde)) return;
            if (!DA.GetData(8, ref e)) return;

            int n = knotsCsi.Count - p - 1;
            int m = knotsEta.Count - q - 1;
            int l = knotsZeta.Count - r - 1;
            int nel = (n - p) * (m - q) * (l - r);
            int nnp = n * m * l;
            int nen = (p + 1) * (q + 1) * (r + 1);

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
            if (e > nel || e < 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Number of element (e) is either too high or below one");
                return;
            }

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<List<List<Point3d>>> B = CreateControlPointList(controlPoints, n, m, l);
            (List<List<int>> INN, List<List<int>> IEN) = CreateINN_IEN(p, q, r, n, m, l);
            (List<int> nCoord, List<double> pCoord) = GetCoordinates(INN, IEN, knotsCsi, knotsEta, knotsZeta, csiTilde, e);
            (List<List<double>> NdN, List<List<double>> MdM, List<List<double>> LdL) = UnivariateBasisFuntion(p, q, r, nCoord, pCoord, knotsCsi, knotsEta, knotsZeta);
            (List<double> R, List<List<double>> dR_dCsi) = TrivariateBasisFunction(p, q, r, NdN, MdM, LdL);
            (Matrix<double> dx_dCsi, Matrix<double> dCsi_dCsiTilde) = CreateGradients(p, q, r, nCoord, B, dR_dCsi, knotsCsi, knotsEta, knotsZeta);
            Matrix<double> J_mat = dx_dCsi * dCsi_dCsiTilde;
            double J = J_mat.Determinant();
            Matrix<double> dCsi_dx = dx_dCsi.Inverse();
            List<List<double>> dR_dx = TrivariateBasisFunctionPhysical(nen, dR_dCsi, dCsi_dx);

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetDataList(0, NdN[0]);
            DA.SetDataList(1, MdM[0]);
            DA.SetDataList(2, LdL[0]);
            DA.SetDataList(3, NdN[1]);
            DA.SetDataList(4, MdM[1]);
            DA.SetDataList(5, LdL[1]);
            DA.SetDataList(6, R);
            DA.SetDataList(7, dR_dCsi[0]);
            DA.SetDataList(8, dR_dCsi[1]);
            DA.SetDataList(9, dR_dCsi[2]);
            DA.SetDataList(10, dR_dx[0]);
            DA.SetDataList(11, dR_dx[1]);
            DA.SetDataList(12, dR_dx[2]);
            DA.SetData(13, dx_dCsi);
            DA.SetData(14, dCsi_dCsiTilde);
            DA.SetData(15, J_mat);
            DA.SetData(16, J);
        }


        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

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

        (List<int>, List<double>) GetCoordinates(List<List<int>> INN, List<List<int>> IEN, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta, List<double> csiTilde, int e)
        {
            List<int> nCoord = new List<int>();
            List<double> pCoord = new List<double>();

            int ni = INN[IEN[e - 1][0] - 1][0];
            double csi = ((knotsCsi[ni] - knotsCsi[ni - 1]) * csiTilde[0] + (knotsCsi[ni] + knotsCsi[ni - 1])) / 2;
            int nj = INN[IEN[e - 1][0] - 1][1];
            double eta = ((knotsEta[nj] - knotsEta[nj - 1]) * csiTilde[1] + (knotsEta[nj] + knotsEta[nj - 1])) / 2;
            int nk = INN[IEN[e - 1][0] - 1][2];
            double zeta = ((knotsZeta[nk] - knotsZeta[nk - 1]) * csiTilde[2] + (knotsZeta[nk] + knotsZeta[nk - 1])) / 2;

            nCoord.Add(ni);
            nCoord.Add(nj);
            nCoord.Add(nk);

            pCoord.Add(csi);
            pCoord.Add(eta);
            pCoord.Add(zeta);

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

            for ( int k = 0; k <= r; k++)
            {
                for ( int j = 0; j <= q; j++)
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
                                dx_dCsi[a,b] += dR_dCsi[b][loc] * pointList[a];
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
            get { return new Guid("77c731df-34ad-4e33-b15e-e37131551e3d"); }
        }
    }
}