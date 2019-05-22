using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;

namespace IGA
{
    public class Create_Box : GH_Component
    {

        public Create_Box()
          : base("Create Box", "create_box",
              "Creates a box, given corner points, mesh dimensions and degrees",
              "IGA", "01. Model")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddPointParameter("Corner points", "cp", "Corner points of box", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Mesh dim", "dim", "Dimensions of mesh [U, V, W]", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Degrees", "deg", "Degrees [p, q, r]", GH_ParamAccess.list);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Patch", "patch", "Patch", GH_ParamAccess.item);
            pManager.AddTextParameter("Info", "info", "Info", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {

            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<Point3d> cp = new List<Point3d>();
            List<int> dim = new List<int>();
            List<int> deg = new List<int>();


            if (!DA.GetDataList(0, cp)) return;
            if (!DA.GetDataList(1, dim)) return;
            if (!DA.GetDataList(2, deg)) return;

            int p = deg[0];
            int q = deg[1];
            int r = deg[2];
            int U = dim[0];
            int V = dim[1];
            int W = dim[2];

            if (cp.Count != 8)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "There are not eight corner points");
                return;
            }

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<double> knotsCsi = CreateKnotvector(p, U);
            List<double> knotsEta = CreateKnotvector(q, V);
            List<double> knotsZeta = CreateKnotvector(r, W);

            int n = knotsCsi.Count - (p + 1);
            int m = knotsEta.Count - (q + 1);
            int l = knotsZeta.Count - (r + 1);
            List<int> nml = new List<int>() { n, m, l };

            List<Point3d > controlPoints = mesh8p(cp, nml);
            List<List<List<Point3d>>> B = CreateControlPointList(controlPoints, n, m, l);
            (List<List<int>> INN, List<List<int>> IEN) = CreateINN_IEN(p, q, r, n, m, l);
            Patch patch = new Patch(B, knotsCsi, knotsEta, knotsZeta, p, q, r, INN, IEN);
            string info = patch.GetMeshInfo();

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, patch);
            DA.SetData(1, info);

        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        List<double> CreateKnotvector(int deg, int dim)
        {
            List<double> knots = new List<double>();

            for (int i = 0; i <= deg; i++)
            {
                knots.Add(0);
            }

            double dDim = dim;
            double step = 1 / dDim;
            for (int i = 1; i < dim; i++)
            {
                knots.Add(i * step);
            }

            for (int i = 0; i <= deg; i++)
            {
                knots.Add(1);
            }

            return knots;
        }

        List<Point3d> mesh2p(Point3d p1, Point3d p2, int dim)
        {

            List<Point3d> Temp_X = new List<Point3d>();


            for (int i = 0; i <= dim; i++)
            {

                double tp1x = p1.X + i * (p2.X - p1.X) / dim;
                double tp1y;
                double tp1z;

                if (p2.X == p1.X)
                {
                    tp1y = p1.Y + i * (p2.Y - p1.Y) / dim;
                    tp1z = p1.Z + i * (p2.Z - p1.Z) / dim;
                }
                else
                {
                    tp1y = (p2.Y - p1.Y) / (p2.X - p1.X) * (tp1x - p1.X) + p1.Y;
                    tp1z = (p2.Z - p1.Z) / (p2.X - p1.X) * (tp1x - p1.X) + p1.Z;
                }

                Temp_X.Add(new Point3d(tp1x, tp1y, tp1z));
            }



            return Temp_X;

        }

        List<Point3d> mesh4p(Point3d p1, Point3d p2, Point3d p3, Point3d p4, int U, int V)
        {

            List<Point3d> List = new List<Point3d>();
            List<Point3d> TempY1 = new List<Point3d>();
            List<Point3d> TempY2 = new List<Point3d>();

            TempY1 = mesh2p(p1, p4, V);
            TempY2 = mesh2p(p2, p3, V);

            for (int i = 0; i < TempY1.Count; i++)
            {
                List.AddRange(mesh2p(TempY1[i], TempY2[i], U));
            }


            return List;
        }

        List<Point3d> mesh8p(List<Point3d> iPoints, List<int> meshDim)
        {
            int U = meshDim[0] - 1;
            int V = meshDim[1] - 1;
            int W = meshDim[2] - 1;

            List<Point3d> TempZ1 = mesh2p(iPoints[0], iPoints[4], W);
            List<Point3d> TempZ2 = mesh2p(iPoints[1], iPoints[5], W);
            List<Point3d> TempZ3 = mesh2p(iPoints[2], iPoints[6], W);
            List<Point3d> TempZ4 = mesh2p(iPoints[3], iPoints[7], W);

            List<Point3d> MeshZ = new List<Point3d>();

            for (int i = 0; i < TempZ1.Count; i++)
            {

                MeshZ.AddRange(mesh4p(TempZ1[i], TempZ2[i], TempZ3[i], TempZ4[i], U, V));

            }

            return MeshZ;
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
            get { return new Guid("a6eb15b0-aa53-44c3-a02c-b93db9d885e5"); }
        }
    }
}