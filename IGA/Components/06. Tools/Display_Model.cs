using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;
using MathNet.Numerics.LinearAlgebra;

namespace IGA.Components._06._Tools
{
    public class Display_Model : GH_Component
    {

        public Display_Model()
          : base("Display Model", "display_model",
              "Display a model with a given grid. If the model is analyzed the displacements can be displayet with a given scale factor",
              "IGA", "06. Tools")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Model", "model", "Model", GH_ParamAccess.item);
            pManager.AddBooleanParameter("True mesh", "btm", "Displays the elements as defined by the knot vectors", GH_ParamAccess.item, false);
            pManager.AddIntegerParameter("Grid", "grid", "Grid dimensions [Csi, Eta, Zeta]", GH_ParamAccess.list);
            pManager.AddNumberParameter("Step", "step", "Step size", GH_ParamAccess.item, 0.01);
            pManager.AddBooleanParameter("Deformed geometry", "bdg", "Displays the deformed geometry if true", GH_ParamAccess.item, false);
            pManager.AddNumberParameter("Scale", "scale", "Scale the deformed geometry", GH_ParamAccess.item, 1);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddLineParameter("Lines", "lines", "Lines representing the geometry", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {

            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            Model model = new Model();
            bool btm = false;
            List<int> grid = new List<int>();
            double step = 0.01;
            bool bdg = false;
            double scale = 1;

            if (!DA.GetData(0, ref model)) return;
            if (!DA.GetData(1, ref btm)) return;
            if (!DA.GetDataList(2, grid)) return;
            if (!DA.GetData(3, ref step)) return;
            if (!DA.GetData(4, ref bdg)) return;
            if (!DA.GetData(5, ref scale)) return;

            if (grid.Count != 3)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "The grid must be specified for three directions");
                return;
            }
            if (step <= 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "The step can not be less or equal to zero");
                return;
            }

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<Line> lines;
            if (btm)
            {
                lines = GetLines(model, step, bdg, scale);
            }
            else
            {
                lines = GetGridLines(model, grid, step, bdg, scale);
            }
            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetDataList(0, lines);
        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        (List<double>, List<double>, List<double>) GetParametricValues(Patch patch)
        {
            List<double> csiValues = new List<double>();
            List<double> etaValues = new List<double>();
            List<double> zetaValues = new List<double>();

            (List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta) = patch.GetKnotVectors();
            (int p, int q, int r) = patch.GetDegrees();
            (int n, int m, int l) = patch.GetNML();

            for (int i = p; i < n; i++)
            {
                if (knotsCsi[i] != knotsCsi[i + 1])
                {
                    csiValues.Add(knotsCsi[i]);
                }
            }
            csiValues.Add(knotsCsi[n]);

            for (int j = q; j < m; j++)
            {
                if (knotsEta[j] != knotsEta[j + 1])
                {
                    etaValues.Add(knotsEta[j]);
                }
            }
            etaValues.Add(knotsEta[m]);

            for (int k = r; k < l; k++)
            {
                if (knotsZeta[k] != knotsZeta[k + 1])
                {
                    zetaValues.Add(knotsZeta[k]);
                }
            }
            zetaValues.Add(knotsZeta[l]);

            return (csiValues, etaValues, zetaValues);
        }

        (List<double>, List<double>, List<double>) GetParametricGridValues(List<int> grid)
        {
            List<double> csiValues = new List<double>();
            List<double> etaValues = new List<double>();
            List<double> zetaValues = new List<double>();

            double csiDim = grid[0];
            double csiStep = 1 / csiDim;
            double etaDim = grid[1];
            double etaStep = 1 / etaDim;
            double zetaDim = grid[2];
            double zetaStep = 1 / zetaDim;

            for (int i = 0; i <= csiDim; i++)
            {
                csiValues.Add(i * csiStep);
            }
            for (int i = 0; i <= etaDim; i++)
            {
                etaValues.Add(i * etaStep);
            }
            for (int i = 0; i <= zetaDim; i++)
            {
                zetaValues.Add(i * zetaStep);
            }

            return (csiValues, etaValues, zetaValues);
        }

        List<double> GetSteps(double knot1, double knot2, double step)
        {
            List<double> steps = new List<double>();

            for (double i = knot1; i < knot2 - step; i += step)
            {
                steps.Add(i);
            }
            steps.Add(knot2);

            return steps;
        }

        List<double> GetGridSteps(double step)
        {
            List<double> steps = new List<double>();

            for (double i = 0; i < 1 - step; i += step)
            {
                steps.Add(i);
            }
            steps.Add(1);

            return steps;
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

        List<List<List<Point3d>>> GetNewControlPoints(List<List<List<Point3d>>> B, Vector<double> d, double scale, int n, int m, int l)
        {
            List<List<List<Point3d>>> newB = new List<List<List<Point3d>>>();

            int id;
            Point3d point;
            for (int k = 0; k < l; k++)
            {
                List<List<Point3d>> newBj = new List<List<Point3d>>();
                for (int j = 0; j < m; j++)
                {
                    List<Point3d> newBi = new List<Point3d>();
                    for (int i = 0; i < n; i++)
                    {
                        id = k * n * m + j * n + i;
                        point = B[k][j][i];
                        newBi.Add(point + new Point3d(scale * d[id * 3], scale * d[id * 3 + 1], scale * d[id * 3 + 2]));
                    }
                    newBj.Add(newBi);
                }
                newB.Add(newBj);
            }

            return newB;
        }

        Point3d CreatePoint(int n, int m, int l, int p, int q, int r, List<List<List<Point3d>>> B, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta, double csi, double eta, double zeta)
        {

            Point3d point = new Point3d(0, 0, 0);
            double N, M, L;

            for (int k = 0; k < l; k++)
            {
                for (int j = 0; j < m; j++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        N = BasisFunction(knotsCsi, i, p, csi);
                        M = BasisFunction(knotsEta, j, q, eta);
                        L = BasisFunction(knotsZeta, k, r, zeta);
                        point += N * M * L * B[k][j][i];
                    }
                }
            }

            return point;
        }

        List<Point3d> GetCsiPoints(int n, int m, int l, int p, int q, int r, List<List<List<Point3d>>> B, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta, List<double> steps, double eta, double zeta)
        {
            List<Point3d> points = new List<Point3d>();


            foreach (double d in steps)
            {
                points.Add(CreatePoint(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, d, eta, zeta));
            }

            return points;
        }

        List<Point3d> GetEtaPoints(int n, int m, int l, int p, int q, int r, List<List<List<Point3d>>> B, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta, double csi, List<double> steps, double zeta)
        {
            List<Point3d> points = new List<Point3d>();


            foreach (double d in steps)
            {
                points.Add(CreatePoint(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, d, zeta));
            }

            return points;
        }

        List<Point3d> GetZetaPoints(int n, int m, int l, int p, int q, int r, List<List<List<Point3d>>> B, List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta, double csi, double eta, List<double> steps)
        {
            List<Point3d> points = new List<Point3d>();


            foreach (double d in steps)
            {
                points.Add(CreatePoint(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, eta, d));
            }

            return points;
        }

        List<Line> GetLinesFromPoints(List<Point3d> points)
        {
            List<Line> lines = new List<Line>();

            for (int i = 0; i < points.Count - 1; i++)
            {
                lines.Add(new Line(points[i], points[i + 1]));
            }

            return lines;
        }

        List<Line> GetLines(Model model, double step, bool bdg, double scale)
        {
            List<Line> lines = new List<Line>();
            List<Line> csiLines = new List<Line>();
            List<Line> etaLines = new List<Line>();
            List<Line> zetaLines = new List<Line>();

            Patch patch = model.GetPatch();
            Vector<double> d = model.GetDisplacementVector();

            List<List<List<Point3d>>> B;
            (List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta) = patch.GetKnotVectors();
            (int p, int q, int r) = patch.GetDegrees();
            (int n, int m, int l) = patch.GetNML();
            (List<double> csiValues, List<double> etaValues, List<double> zetaValues) = GetParametricValues(patch);

            if (bdg)
            {
                List<List<List<Point3d>>> oB = patch.GetControlPoints();
                B = GetNewControlPoints(oB, d, scale, n, m, l);
            }
            else
            {
                B = patch.GetControlPoints();
            }

            double zeta, eta, csi;
            List<double> zetaSteps, etaSteps, csiSteps;
            List<Point3d> csiPoints, etaPoints, zetaPoints;
            for (int k = 0; k < zetaValues.Count; k++)
            {
                zeta = zetaValues[k];

                for (int j = 0; j < etaValues.Count; j++)
                {
                    eta = etaValues[j];

                    for (int i = 0; i < csiValues.Count; i++)
                    {
                        csi = csiValues[i];

                        if (i == csiValues.Count - 1 && j == etaValues.Count - 1 && k == zetaValues.Count - 1)
                        {

                        }
                        else if (i == csiValues.Count - 1 && k == zetaValues.Count - 1)
                        {
                            etaSteps = GetSteps(eta, etaValues[j + 1], step);
                            etaPoints = GetEtaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, etaSteps, zeta);
                            etaLines.AddRange(GetLinesFromPoints(etaPoints));
                        }
                        else if (j == etaValues.Count - 1 && k == zetaValues.Count - 1)
                        {
                            csiSteps = GetSteps(csi, csiValues[i + 1], step);
                            csiPoints = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csiSteps, eta, zeta);
                            csiLines.AddRange(GetLinesFromPoints(csiPoints));
                        }
                        else if (k == zetaValues.Count - 1)
                        {
                            csiSteps = GetSteps(csi, csiValues[i + 1], step);
                            etaSteps = GetSteps(eta, etaValues[j + 1], step);
                            csiPoints = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csiSteps, eta, zeta);
                            csiLines.AddRange(GetLinesFromPoints(csiPoints));
                            etaPoints = GetEtaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, etaSteps, zeta);
                            etaLines.AddRange(GetLinesFromPoints(etaPoints));
                        }
                        else if (i == csiValues.Count - 1 && j == etaValues.Count - 1)
                        {
                            zetaSteps = GetSteps(zeta, zetaValues[k + 1], step);
                            zetaPoints = GetZetaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, eta, zetaSteps);
                            zetaLines.AddRange(GetLinesFromPoints(zetaPoints));
                        }
                        else if (i == csiValues.Count - 1)
                        {
                            etaSteps = GetSteps(eta, etaValues[j + 1], step);
                            zetaSteps = GetSteps(zeta, zetaValues[k + 1], step);
                            etaPoints = GetEtaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, etaSteps, zeta);
                            etaLines.AddRange(GetLinesFromPoints(etaPoints));
                            zetaPoints = GetZetaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, eta, zetaSteps);
                            zetaLines.AddRange(GetLinesFromPoints(zetaPoints));
                        }
                        else if (j == etaValues.Count - 1)
                        {
                            csiSteps = GetSteps(csi, csiValues[i + 1], step);
                            zetaSteps = GetSteps(zeta, zetaValues[k + 1], step);
                            csiPoints = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csiSteps, eta, zeta);
                            csiLines.AddRange(GetLinesFromPoints(csiPoints));
                            zetaPoints = GetZetaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, eta, zetaSteps);
                            zetaLines.AddRange(GetLinesFromPoints(zetaPoints));

                        }
                        else
                        {
                            csiSteps = GetSteps(csi, csiValues[i + 1], step);
                            etaSteps = GetSteps(eta, etaValues[j + 1], step);
                            zetaSteps = GetSteps(zeta, zetaValues[k + 1], step);
                            csiPoints = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csiSteps, eta, zeta);
                            csiLines.AddRange(GetLinesFromPoints(csiPoints));
                            etaPoints = GetEtaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, etaSteps, zeta);
                            etaLines.AddRange(GetLinesFromPoints(etaPoints));
                            zetaPoints = GetZetaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, eta, zetaSteps);
                            zetaLines.AddRange(GetLinesFromPoints(zetaPoints));
                        }
                    }
                }
            }

            lines.AddRange(csiLines);
            lines.AddRange(etaLines);
            lines.AddRange(zetaLines);

            return lines;
        }

        List<Line> GetGridLines(Model model, List<int> grid, double step, bool bdg, double scale)
        {
            List<Line> lines = new List<Line>();
            List<Line> csiLines = new List<Line>();
            List<Line> etaLines = new List<Line>();
            List<Line> zetaLines = new List<Line>();

            Patch patch = model.GetPatch();
            Vector<double> d = model.GetDisplacementVector();

            List<List<List<Point3d>>> B = patch.GetControlPoints();
            (List<double> knotsCsi, List<double> knotsEta, List<double> knotsZeta) = patch.GetKnotVectors();
            (int p, int q, int r) = patch.GetDegrees();
            (int n, int m, int l) = patch.GetNML();
            (List<double> csiValues, List<double> etaValues, List<double> zetaValues) = GetParametricGridValues(grid);
            List<double> steps = GetGridSteps(step);

            if (bdg)
            {
                List<List<List<Point3d>>> oB = patch.GetControlPoints();
                B = GetNewControlPoints(oB, d, scale, n, m, l);
            }
            else
            {
                B = patch.GetControlPoints();
            }

            double csi;
            for (int i = 0; i < csiValues.Count; i++)
            {
                csi = csiValues[i];
                List<Point3d> eta_zeta0 = GetEtaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, steps, 0);
                List<Line> eta0 = GetLinesFromPoints(eta_zeta0);
                List<Point3d> eta_zeta1 = GetEtaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, steps, 1);
                List<Line> eta1 = GetLinesFromPoints(eta_zeta1);
                List<Point3d> zeta_eta0 = GetZetaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, 0, steps);
                List<Line> zeta0 = GetLinesFromPoints(zeta_eta0);
                List<Point3d> zeta_eta1 = GetZetaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, csi, 1, steps);
                List<Line> zeta1 = GetLinesFromPoints(zeta_eta1);
                etaLines.AddRange(eta0);
                etaLines.AddRange(eta1);
                zetaLines.AddRange(zeta0);
                zetaLines.AddRange(zeta1);
            }

            double eta = 0;
            List<Point3d> csi_zeta0_eta0 = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, steps, eta, 0);
            List<Line> csi0_eta0 = GetLinesFromPoints(csi_zeta0_eta0);
            List<Point3d> csi_zeta1_eta0 = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, steps, eta, 1);
            List<Line> csi1_eta0 = GetLinesFromPoints(csi_zeta1_eta0);
            csiLines.AddRange(csi0_eta0);
            csiLines.AddRange(csi1_eta0);

            for (int i = 1; i < etaValues.Count - 1; i++)
            {
                eta = etaValues[i];
                List<Point3d> csi_zeta0 = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, steps, eta, 0);
                List<Line> csi0 = GetLinesFromPoints(csi_zeta0);
                List<Point3d> csi_zeta1 = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, steps, eta, 1);
                List<Line> csi1 = GetLinesFromPoints(csi_zeta1);
                List<Point3d> zeta_csi0 = GetZetaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, 0, eta, steps);
                List<Line> zeta0 = GetLinesFromPoints(zeta_csi0);
                List<Point3d> zeta_csi1 = GetZetaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, 1, eta, steps);
                List<Line> zeta1 = GetLinesFromPoints(zeta_csi1);
                csiLines.AddRange(csi0);
                csiLines.AddRange(csi1);
                zetaLines.AddRange(zeta0);
                zetaLines.AddRange(zeta1);
            }

            eta = 1;
            List<Point3d> csi_zeta0_eta1 = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, steps, eta, 0);
            List<Line> csi0_eta1 = GetLinesFromPoints(csi_zeta0_eta1);
            List<Point3d> csi_zeta1_eta1 = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, steps, eta, 1);
            List<Line> csi1_eta1 = GetLinesFromPoints(csi_zeta1_eta1);
            csiLines.AddRange(csi0_eta1);
            csiLines.AddRange(csi1_eta1);

            double zeta;
            for (int i = 1; i < zetaValues.Count - 1; i++)
            {
                zeta = zetaValues[i];
                List<Point3d> csi_eta0 = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, steps, 0, zeta);
                List<Line> csi0 = GetLinesFromPoints(csi_eta0);
                List<Point3d> csi_eta1 = GetCsiPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, steps, 1, zeta);
                List<Line> csi1 = GetLinesFromPoints(csi_eta1);
                List<Point3d> eta_csi0 = GetEtaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, 0, steps, zeta);
                List<Line> eta0 = GetLinesFromPoints(eta_csi0);
                List<Point3d> eta_csi1 = GetEtaPoints(n, m, l, p, q, r, B, knotsCsi, knotsEta, knotsZeta, 1, steps, zeta);
                List<Line> eta1 = GetLinesFromPoints(eta_csi1);
                csiLines.AddRange(csi0);
                csiLines.AddRange(csi1);
                etaLines.AddRange(eta0);
                etaLines.AddRange(eta1);
            }

            lines.AddRange(csiLines);
            lines.AddRange(etaLines);
            lines.AddRange(zetaLines);

            return lines;
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
            get { return new Guid("82ff4786-e296-4d89-bc31-7000cfc66ed0"); }
        }
    }
}