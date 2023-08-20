from tqdm import tqdm
from typing import NoReturn
from explorer import create_explorer
from dse_problem import create_dse_problem
from copy import deepcopy


def auger(idx: int = 0):
	problem = create_dse_problem("./dataset/TRAM.csv")
	explorer = create_explorer(problem)

	explorer.initialize()
	
	explorer_mix = deepcopy(explorer)
	time_mix = 0
	iterator = tqdm(range(24))
	for step in iterator:
		iterator.set_description("Iter {}".format(step + 1))
		explorer_mix.fit_and_suggest()
		# solver.eipv_suggest()
	preAdrs_mix, finalAdrs_mix, preHv_mix, finalHv_mix = explorer_mix.report_result("mix_" + str(idx))

	# solver_rf = deepcopy(solver)
	# iterator = tqdm(range(configs["bo"]["max-bo-steps"]))
	# for step in iterator:
	# 	iterator.set_description("Iter {}".format(step + 1))
	# 	solver_rf.fit_RdnForest()
	# 	solver_rf.rf_suggest()
	# 	# solver.eipv_suggest()
	# preAdrs_rf, finalAdrs_rf, preHv_rf, finalHv_rf = solver_rf.report("rf_" + str(idx))

	

	# solver_ehvi = deepcopy(solver)
	# iterator = tqdm(range(configs["bo"]["max-bo-steps"]))
	# for step in iterator:
	# 	iterator.set_description("Iter {}".format(step + 1))
	# 	# solver.fit_and_suggest()
	# 	solver_ehvi.fit_dkl_gp()
	# 	solver_ehvi.eipv_suggest()
	# 	# solver.eipv_suggest()
	# preAdrs_ehvi, finalAdrs_ehvi, preHv_ehvi, finalHv_ehvi = solver_ehvi.report("gpo_" + str(idx))

	# # solver_qehvi = deepcopy(solver)
	# # iterator = tqdm(range(configs["bo"]["max-bo-steps"]))
	# # for step in iterator:
	# # 	iterator.set_description("Iter {}".format(step + 1))
	# # 	# solver.fit_and_suggest()
	# # 	solver_qehvi.fit_dkl_gp()
	# # 	solver_qehvi.qeipv_suggest()
	# # preAdrs_qehvi, finalAdrs_qehvi, preHv_qehvi, finalHv_qehvi = solver_qehvi.report("gp_" + str(idx))

	# # print("EHVI: ", preAdrs_ehvi, " ", finalAdrs_ehvi, " ", preHv_ehvi, " ", finalHv_ehvi)
	# # print("QEHVI: ", preAdrs_qehvi, " ", finalAdrs_qehvi, " ", preHv_qehvi, " ", finalHv_qehvi)
	# print("RF: ", preAdrs_rf, " ", finalAdrs_rf, " ", preHv_rf, " ", finalHv_rf)
	print("MIX: ", preAdrs_mix, " ", finalAdrs_mix, " ", preHv_mix, " ", finalHv_mix)

if __name__ == "__main__":
	auger(0)