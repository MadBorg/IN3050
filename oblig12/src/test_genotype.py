import genotype as g
import main
import data


N = 100
df = data.data(main.path_to_datafile)
evaluator = main.fit
subset_size = 6
rep = [i for i in range(0, 6)]

objs = []
for i in range(N):
    obj = g.Genotype(evaluator)
    obj.make_permutation(rep)


def test_genotype_init():
    # Init

    # Creating objects        
    for obj in objs:
        for element in obj.r:
            try:
                rep.index(element)
            except ValueError:
                raise ValueError("test_genotype failed!")

def test_genotype_score():

    # Calculating scores
    for obj in objs:
        obj.calculate_score(df.get_subset(subset_size))
        tmp_score = obj.score
        print(f"Score: {tmp_score}")
        # assert type(tmp_score) is float, f"Score is not int, score is {type(tmp_score)}"
        assert tmp_score > 0, f"Score is not larger then 0, score is {tmp_score}"
    
if __name__ == "__main__":
    test_genotype_init()
    test_genotype_score()