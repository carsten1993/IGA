using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace IGA.Components._05._Analyze
{
    public class Analyze_Model : GH_Component
    {

        public Analyze_Model()
          : base("Analyze Model", "analyze_model",
              "Analyze a model",
              "IGA", "05. Analyze")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Model", "model", "Model to analyze", GH_ParamAccess.item);
            pManager.AddBooleanParameter("Self weight", "self_weight", "True to include self weight", GH_ParamAccess.item, false);
            //pManager.AddBooleanParameter("Strain and stress", "strain_stress", "True to calculate strain and stress", GH_ParamAccess.item, false);
            //pManager.AddBooleanParameter("Mises", "mises", "True to calculate Mises", GH_ParamAccess.item, false);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Analyzed model", "analyzed_model", "Analyzed model", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            Model model = new Model();
            bool selfWeight = false;
            //bool strainStress = false;
            //bool mises = false;

            if (!DA.GetData(0, ref model)) return;
            if (!DA.GetData(1, ref selfWeight)) return;
            //if (!DA.GetData(2, ref strainStress)) return;
            //if (!DA.GetData(3, ref mises)) return;
            /*
            if (!strainStress && mises)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Can not calculate Mises without calculating the strain and stress");
                return;
            }*/

            Patch patch = model.GetPatch();
            Material material = model.GetMaterial();
            //List<PointLoad> pointLoads = model.GetPointLoads();
            List<Restraint> restraints = model.GetRestraints();
            List<List<int>> res = model.GetCompleteRestraints();
            Vector<double> P0 = model.GetSelfLoadVector();
            Vector<double> P1 = model.GetExternalLoadVector();
            Matrix<double> K_full = model.GetFullStiffnessMatrix();

            (int nel, int nnp, int nen) = patch.GetSize();
            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            Vector<double> P_full = P1;
            if (selfWeight)
            {
                P_full += P0;
            }            

            (Matrix<double> K, Vector<double> P) = ReshapeStiffnessMatrixLoadVector(K_full, P_full, res, nnp);
            Vector<double> d = K.Inverse() * P;
            patch.SetNewControlPoints(d);

            //Model newModel = new Model(mesh, material, pointLoads, restraints);
            Model newModel = new Model(patch, material, restraints);
            newModel.SetCompleteRestraints(res);
            newModel.SetExternalLoadVector(P1);
            newModel.SetSelfLoadVector(P0);
            newModel.SetLoadVector(P);
            newModel.SetDisplacementVector(d);
            newModel.SetFullStiffnessMatrix(K_full);
            newModel.SetStiffnessMatrix(K);

            /*
            if (strainStress)
            {
                (Matrix<double> strains, Matrix<double> stresses) = StrainsStresses(elements, r, C);
                (List<double> strainList, List<double> stressList) = MatrixToList_double(strains, stresses);
                newModel.SetStrains(strainList);
                newModel.SetStresses(stressList);

                if (mises)
                {
                    List<double> misesList = Mises(stresses);
                    newModel.SetMises(misesList);
                }
            }*/

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, newModel);
        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        (Matrix<double>, Vector<double>) ReshapeStiffnessMatrixLoadVector(Matrix<double> K_full, Vector<double> P_full, List<List<int>> res, int nnp)
        {
            Matrix<double> K = DenseMatrix.OfMatrix(K_full);
            Vector<double> P = DenseVector.OfVector(P_full);

            int id, resx, resy, resz;
            foreach (List<int> list in res)
            {
                id = 3 * (list[0] - 1);
                resx = list[1];
                resy = list[2];
                resz = list[3];

                if (resx == 1)
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
            get { return new Guid("76489882-7c35-4477-b109-26f785bb1237"); }
        }
    }
}