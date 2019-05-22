using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace IGA.Components._01._Model
{
    public class Assemble_Model : GH_Component
    {

        public Assemble_Model()
          : base("Assemble Model", "assemble_model",
              "Construct a model ready for analysis, given a patch, material properties, restraints and loads.",
              "IGA", "01. Model")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Patch", "patch", "Patch", GH_ParamAccess.item);
            pManager.AddGenericParameter("Material", "material", "Material properties", GH_ParamAccess.item);
            //pManager.AddGenericParameter("Loads", "loads", "External loads", GH_ParamAccess.list);
            pManager.AddGenericParameter("Restraints", "restraints", "Nodal restraints", GH_ParamAccess.list);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Model", "model", "Model ready for analysis", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            Patch patch = new Patch();
            Material material = new Material();
            //List<PointLoad> pointLoads = new List<PointLoad>();
            List<Restraint> restraints = new List<Restraint>();

            if (!DA.GetData(0, ref patch)) return;
            if (!DA.GetData(1, ref material)) return;
            //if (!DA.GetDataList(2, pointLoads)) return;
            if (!DA.GetDataList(2, restraints)) return;

            (int nel, int nnp, int nen) = patch.GetSize();

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<List<int>> res = CreateRestraintList(restraints);
            Vector<double> P1 = CreateExternalLoadVector(patch);
            (Matrix<double> K_full, Vector<double> P0) = CreateStiffnessMatrixLoadVector(patch, material);

            //Model model = new Model(mesh, material, pointLoads, restraints);
            Model model = new Model(patch, material, restraints);
            model.SetCompleteRestraints(res);
            model.SetExternalLoadVector(P1);
            model.SetSelfLoadVector(P0);
            model.SetFullStiffnessMatrix(K_full);

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////


            DA.SetData(0, model);
        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        List<List<int>> CreateRestraintList(List<Restraint> restraints)
        {
            List<List<int>> res = new List<List<int>>();

            List<List<int>> res0 = restraints[0].GetRestraints();
            foreach (List<int> l in res0)
            {
                res.Add(l);
            }

            List<List<int>> resTemp;
            int c;
            for (int i = 1; i < restraints.Count; i++)
            {
                resTemp = restraints[i].GetRestraints();
                foreach (List<int> l in resTemp)
                {
                    c = 0;
                    for (int j = 0; j < res.Count; j++)
                    {
                        if (l[0] == res[j][0])
                        {
                            if (l[1] == 1 && res[j][1] == 0)
                            {
                                res[j][1] = 1;
                            }
                            if (l[2] == 1 && res[j][2] == 0)
                            {
                                res[j][2] = 1;
                            }
                            if (l[3] == 1 && res[j][3] == 0)
                            {
                                res[j][3] = 1;
                            }
                            c++;
                        }
                    }
                    if (c == 0)
                    {
                        res.Add(l);
                    }
                }
            }

            return res;
        }

        Vector<double> CreateExternalLoadVector(Patch patch)
        {
            (int n, int m, int l) = patch.GetNML();
            (int nel, int nnp, int nen) = patch.GetSize();
            Vector<double> P = DenseVector.OfArray(new double[3 * nnp]);
            List<int> p = patch.GetSurface3();
            /*
            foreach(int d in p)
            {
                P[(d - 1) * 3 + 2] = -1000;
            }*/

            return P;
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

        List<int> GetNURBSCoord(List<List<int>> INN, List<List<int>> IEN, int e)
        {
            int ni = INN[IEN[e - 1][0] - 1][0];
            int nj = INN[IEN[e - 1][0] - 1][1];
            int nk = INN[IEN[e - 1][0] - 1][2];

            return new List<int>() { ni, nj, nk };
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

        Matrix<double> CreateGaussPointElementStiffnessMatrix(List<List<double>> dR_dx, int nen, Matrix<double> C, double modJ)
        {
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

        (Matrix<double>, Vector<double>) CreateElementStiffnessMatrixLoadVector(int p, int q, int r, int nen, List<int> nCoord, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta, List<List<List<Point3d>>> B, Matrix<double> C, double df)
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
                        Ke += CreateGaussPointElementStiffnessMatrix(dR_dx, nen, C, modJ);
                        Pe += CreateGaussPointElementLoadVector(R, nen, df, modJ);
                    }
                }
            }

            return (Ke, Pe);
        }

        (Matrix<double>, Vector<double>) CreateStiffnessMatrixLoadVector(Patch patch, Material material)
        {
            List<List<List<Point3d>>> B = patch.GetControlPoints();
            (List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta) = patch.GetKnotVectors();
            (int p, int q, int r) = patch.GetDegrees();
            (int n, int m, int l) = patch.GetNML();
            (int nel, int nnp, int nen) = patch.GetSize();
            (List<List<int>> INN, List<List<int>> IEN) = patch.GetINNIEN();
            double E_Y = material.GetEmodulus();
            double v_P = material.GetPoission();
            double df = material.GetDensityForce();
            Matrix<double> C = material.GetCmatrix();

            double lambda = v_P * E_Y / ((1 + v_P) * (1 - 2 * v_P));
            double mu = E_Y / (2 * (1 + v_P));

            

            Matrix<double> K = DenseMatrix.OfArray(new double[3 * nnp, 3 * nnp]);
            Matrix<double> Ke = DenseMatrix.OfArray(new double[3 * nen, 3 * nen]);
            Vector<double> P0 = DenseVector.OfArray(new double[3 * nnp]);
            Vector<double> P0e = DenseVector.OfArray(new double[3 * nen]);

            for (int e = 1; e <= nel; e++)
            {
                List<int> nCoord = GetNURBSCoord(INN, IEN, e);
                int ni = nCoord[0];
                int nj = nCoord[1];
                int nk = nCoord[2];

                if (knotsCsi[ni] == knotsCsi[ni - 1] || knotsEta[nj] == knotsEta[nj - 1] || knotsZeta[nk] == knotsZeta[nk - 1])
                {
                    continue;
                }

                (Ke, P0e) = CreateElementStiffnessMatrixLoadVector(p, q, r, nen, nCoord, knotsCsi, knotsEta, knotsZeta, B, C, df);

                int localI1, localI2, localI3, localJ1, localJ2, localJ3, globalI1, globalI2, globalI3, globalJ1, globalJ2, globalJ3, globalFI, globalFJ;
                for (int i = 0; i < 3 * nen; i += 3)
                {
                    globalFI = nen - 1 - (i / 3);

                    localI1 = i;
                    localI2 = i + 1;
                    localI3 = i + 2;

                    globalI1 = (IEN[e - 1][globalFI] - 1) * 3;
                    globalI2 = (IEN[e - 1][globalFI] - 1) * 3 + 1;
                    globalI3 = (IEN[e - 1][globalFI] - 1) * 3 + 2;

                    P0[globalI1] += P0e[localI1];
                    P0[globalI2] += P0e[localI2];
                    P0[globalI3] += P0e[localI3];

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

            return (K, P0);

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
            get { return new Guid("28774200-6a9c-4a13-9874-ac7951f4fc19"); }
        }
    }
}