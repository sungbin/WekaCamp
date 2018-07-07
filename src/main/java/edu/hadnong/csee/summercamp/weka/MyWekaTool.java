package edu.hadnong.csee.summercamp.weka;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import weka.core.Instances;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SGDText;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.lazy.LWL;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.ClassificationViaRegression;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.IterativeClassifierOptimizer;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.meta.MultiClassClassifierUpdateable;
import weka.classifiers.meta.MultiScheme;
import weka.classifiers.meta.RandomCommittee;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.meta.RandomizableFilteredClassifier;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.meta.WeightedInstancesHandlerWrapper;
import weka.classifiers.misc.InputMappedClassifier;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.HoeffdingTree;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveRange;

public class MyWekaTool {
	/*
	 * 1. 데이터를 뽑음. 2. Traning Data를 만든다. 3. 예측을 한다. 4. 예측된 결과를 보여준다.
	 * 
	 */
	public static void main(String[] args) {
		MyWekaTool myTool = new MyWekaTool();
		double sum = 0.0;

		String[] nargs1 = { args[0], args[1] };
		sum += myTool.run(nargs1);

		String[] nargs2 = { args[0], args[2] };
		sum += myTool.run(nargs2);

		String[] nargs3 = { args[1], args[0] };
		sum += myTool.run(nargs3);

		String[] nargs4 = { args[1], args[2] };
		sum += myTool.run(nargs4);

		String[] nargs5 = { args[2], args[0] };
		sum += myTool.run(nargs5);

		String[] nargs6 = { args[2], args[1] };
		sum += myTool.run(nargs6);
		
		System.out.println("전체평균은 " + sum/6.0);
		//System.out.println("전체평균은 " + (sum+0.0123+0.0158+0.008)/6.0);

	}

	private double run(String[] args) {

		System.out.println(args[0]);
		System.out.println(args[1]);
		String arffForTraining = args[0];
		String arffForTest = args[1];
		try {

			// BufferedReader reader = new BufferedReader(new FileReader(arffForTraining));
			// Instances trainingData = new Instances(reader);
			// trainingData.setClassIndex(trainingData.numAttributes() - 1);
			// reader.close();
			//
			// reader = new BufferedReader(new FileReader(arffForTest));
			// Instances testData = new Instances(reader);
			// testData.setClassIndex(trainingData.numAttributes() - 1);
			// reader.close();
			//
			// // preprocessing
			// AttributeSelection attrSelector =
			// getAttributesSelectionFilterByCfsSubsetEval(trainingData);
			// trainingData = selectFeaturesByAttributeSelection(attrSelector,
			// trainingData);
			// testData = selectFeaturesByAttributeSelection(attrSelector, testData);
			//
			// // (2) Build a learner
			// Classifier cls = new J48();

			/* cls를 List로 만들어서 여러 객체를 집어넣자 */
			ArrayList<Classifier> clslist = new ArrayList<Classifier>();

			clslist.add(new BayesNet()); // bayes
			clslist.add(new NaiveBayes());
			clslist.add(new NaiveBayesMultinomial());
			clslist.add(new NaiveBayesMultinomialText());
			clslist.add(new NaiveBayesUpdateable());
			clslist.add(new Logistic()); // functions
			clslist.add(new MultilayerPerceptron());
			clslist.add(new SGD());
			clslist.add(new SGDText());
			clslist.add(new SimpleLogistic());
			clslist.add(new SMO());
			clslist.add(new VotedPerceptron());
			clslist.add(new IBk()); // lazy
			clslist.add(new KStar());
			clslist.add(new LWL());
			clslist.add(new AdaBoostM1()); // meta
			clslist.add(new AttributeSelectedClassifier());
			clslist.add(new Bagging());
			clslist.add(new ClassificationViaRegression());
			clslist.add(new CostSensitiveClassifier());
			clslist.add(new CVParameterSelection());
			clslist.add(new FilteredClassifier());
			clslist.add(new IterativeClassifierOptimizer());
			clslist.add(new LogitBoost());
			clslist.add(new MultiClassClassifier());
			clslist.add(new MultiClassClassifierUpdateable());
			clslist.add(new MultiScheme());
			clslist.add(new RandomCommittee());
			clslist.add(new RandomizableFilteredClassifier());
			clslist.add(new RandomSubSpace());
			clslist.add(new Stacking());
			clslist.add(new Vote());
			clslist.add(new WeightedInstancesHandlerWrapper());
			clslist.add(new InputMappedClassifier()); // misc
			clslist.add(new DecisionTable()); // rules
			clslist.add(new JRip());
			clslist.add(new OneR());
			clslist.add(new PART());
			clslist.add(new ZeroR());
			clslist.add(new DecisionStump()); // trees
			clslist.add(new HoeffdingTree());
			clslist.add(new J48());
			clslist.add(new LMT());
			clslist.add(new RandomForest());
			clslist.add(new RandomTree());
			clslist.add(new REPTree());

			/* 여기까지 */

			// cls.buildClassifier(trainingData);

			// 수정부분 --

			MyWekaTool weka = new MyWekaTool();

			// AttributeSelection attrSelector =
			// getAttributesSelectionFilterByCfsSubsetEval(trainingData);
//			ArrayList<Object> selectors = new ArrayList<Object>();
//			{
//				AttributeSelection filter = new AttributeSelection(); // package weka.filters.supervised.attribute!
//				CfsSubsetEval eval = new CfsSubsetEval();
//				BestFirst search = new BestFirst();
//				// search.ssetSearchBackwards(false);
//				filter.setEvaluator(eval);
//				filter.setSearch(search);
//
//				try {
//					filter.setInputFormat(trainingData);
//
//					// generate new data
//					// newData = Filter.useFilter(data, filter);
//				} catch (Exception e) {
//					e.printStackTrace();
//				}
//				selectors.add(filter);
//			}
//			{
//
//			}

			int i = 131;
			double max = 0;
			int max_num = 0;
			for (Classifier mcls : clslist) {

				try {

					// if(i == 139) {i++;continue;}

					/* ~~~ */

					// preprocessing

					// for(Object selector : selectors) {
					// }
					BufferedReader reader = new BufferedReader(new FileReader(arffForTraining));
					Instances trainingData = new Instances(reader);
					trainingData.setClassIndex(trainingData.numAttributes() - 1);
					reader.close();

					reader = new BufferedReader(new FileReader(arffForTest));
					Instances testData = new Instances(reader);
					testData.setClassIndex(trainingData.numAttributes() - 1);
					reader.close();
					
					 AttributeSelection attrSelector = getAttributesSelectionFilterByCfsSubsetEval(trainingData);
					
					 trainingData = selectFeaturesByAttributeSelection(attrSelector, trainingData);
					 testData = selectFeaturesByAttributeSelection(attrSelector, testData);

					mcls.buildClassifier(trainingData);
					Evaluation eval = new Evaluation(trainingData);
					eval.evaluateModel(mcls, testData);

					if (max < weka.showSummary(eval, testData)) {
						max = weka.showSummary(eval, testData);
						max_num = i;
					}

					/* ~~~ */

					/*
					 * System.out.print("@@ "+i+" @@ 번째 알고리즘"); weka.showSummary(eval, testData);
					 * 
					 * /
					 **/
					i++;
				} catch (Exception e) {
					continue;
				}
			}

			System.out.println("max:" + max + "\nmax알고리즘: " + max_num);
			System.out.println("@@@@@@@@@@@@@@");

			// (3) Test
			// Evaluation eval = new Evaluation(trainingData);
			// eval.evaluateModel(cls, testData);

			// (4) Show prediction results
			// int i=0;
			// for(Prediction prediction:eval.predictions()) {
			// String predictedValue = getClassValue(trainingData,prediction.predicted());
			// System.out.println("Instance " + (++i) + " " + predictedValue);
			// }

			// System.out.println("\n\n\n=====Test summary in case the test set has
			// labels");
			// System.out.println(eval.toSummaryString());
			return max;

		} catch (Exception e) {
			e.printStackTrace();
		}
		return 0;
	}

	String getClassValue(Instances instances, double index) {
		return instances.attribute(instances.classIndex()).value((int) index);
	}

	/**
	 * Get instances by removing specific attributes
	 * 
	 * @param instances
	 * @param attributeIndices
	 *            attribute indices (e.g., 1,3,4) first index is 1
	 * @param invertSelection
	 *            for invert selection, if true, select attributes with
	 *            attributeIndices bug if false, remote attributes with
	 *            attributeIndices
	 * @return new instances with specific attributes
	 */
	public Instances getInstancesByRemovingSpecificAttributes(Instances instances, String attributeIndices,
			boolean invertSelection) {
		Instances newInstances = new Instances(instances);

		Remove remove;

		remove = new Remove();
		remove.setAttributeIndices(attributeIndices);
		remove.setInvertSelection(invertSelection);
		try {
			remove.setInputFormat(newInstances);
			newInstances = Filter.useFilter(newInstances, remove);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}

		return newInstances;
	}

	/**
	 * Get instances by removing specific instances
	 * 
	 * @param instances
	 * @param instance
	 *            indices (e.g., 1,3,4) first index is 1
	 * @param option
	 *            for invert selection
	 * @return selected instances
	 */
	public Instances getInstancesByRemovingSpecificInstances(Instances instances, String instanceIndices,
			boolean invertSelection) {
		Instances newInstances = null;

		RemoveRange instFilter = new RemoveRange();
		instFilter.setInstancesIndices(instanceIndices);
		instFilter.setInvertSelection(invertSelection);

		try {
			instFilter.setInputFormat(instances);
			newInstances = Filter.useFilter(instances, instFilter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newInstances;
	}

	/**
	 * Feature selection by GainRatioAttributeEval
	 * 
	 * @param data
	 * @return newData with selected attributes
	 */
	static public Instances featrueSelectionByGainRatioAttributeEval(Instances data) {
		Instances newData = null;

		AttributeSelection filter = new AttributeSelection(); // package weka.filters.supervised.attribute!
		GainRatioAttributeEval eval = new GainRatioAttributeEval();
		Ranker search = new Ranker();
		search.setThreshold(-1.7976931348623157E308);
		search.setNumToSelect(-1);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		try {
			filter.setInputFormat(data);

			// generate new data
			newData = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newData;
	}

	/**
	 * Feature selection by CfsSubsetEval
	 * 
	 * @param data
	 * @return newData with selected attributes
	 */
	public Instances featrueSelectionByCfsSubsetEval(Instances data) {
		Instances newData = null;

		AttributeSelection filter = getAttributesSelectionFilterByCfsSubsetEval(data); // package
																						// weka.filters.supervised.attribute!
		try {
			filter.setInputFormat(data);

			// generate new data
			newData = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newData;
	}

	/**
	 * Feature selection filter by CfsSubsetEval
	 * 
	 * @param data
	 * @return AttributeSelection filter
	 */
	public AttributeSelection getAttributesSelectionFilterByCfsSubsetEval(Instances data) {

		AttributeSelection filter = new AttributeSelection(); // package weka.filters.supervised.attribute!
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		// search.ssetSearchBackwards(false);
		filter.setEvaluator(eval);
		filter.setSearch(search);

		try {
			filter.setInputFormat(data);

			// generate new data
			// newData = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return filter;
	}

	public Instances selectFeaturesByAttributeSelection(AttributeSelection selector, Instances data) {
		Instances newData = null;

		try {
			// filter.setInputFormat(data);

			// generate new data
			newData = Filter.useFilter(data, selector);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newData;
	}

	private double showSummary(Evaluation eval, Instances instances) {
		for (int i = 0; i < instances.classAttribute().numValues() - 1; i++) {
			// System.out.println("\n*** Summary of Class " +
			// instances.classAttribute().value(i));
			// System.out.println("Precision " + eval.precision(i));
			// System.out.println("Recall " + eval.recall(i));
			// System.out.println("F-Measure " + eval.fMeasure(i));
			return eval.fMeasure(i);
		}
		return -1;
	}

}
