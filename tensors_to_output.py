import test_int

def init(d_num):
  all = test_int.torch.Tensor.tolist(test_int.tensors(d_num)[0])

  sorted_list = sorted(all, key=lambda x: x[0])

  output_string = ""


  for i in sorted_list:
    cls = int(i[-1])
    if cls <= 9:
      output_string += str(cls)

    elif cls == 10:
        output_string += "+"

    elif cls == 11:
      output_string += "-"

    elif cls == 12:
      output_string += "*"

    elif cls == 13:
      output_string += "/"

  try:
    output_value =eval(output_string)

  except SyntaxError:
    return(f"sorry, try again I read your input as: {output_string}")

  except ZeroDivisionError:
    return (f"You divided by Zero! ")

  return(f"{output_string} = {output_value}")
